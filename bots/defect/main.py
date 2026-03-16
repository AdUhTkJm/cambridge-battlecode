from cambc import Controller, Direction, EntityType, Environment, Position, GameConstants, GameError
import math
import heapq
import sys
import time

# Phases of the game.
PHASE_BEGIN = 0

# The number of bots that the core should initally attempt to spawn.
PHASE_BEGIN_CORE_BOTS = 6

# Diagonal and straight directions.
DIRECTIONS_DIAG = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
DIRECTIONS_STR = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIRECTIONS = [*DIRECTIONS_DIAG, *DIRECTIONS_STR]

# Bot states.
BOT_UNASSIGNED = 0
BOT_EXPLORER = 1
BOT_CONVEYOR_BUILDER = 2
BOT_WAIT_FOR_HARVEST = 3

def get_id_at(ct: Controller, pos: Position) -> int | None:
  """
  Get entity id at the given position, `pos`.
  """
  id = ct.get_tile_building_id(pos)
  if id:
    return id
  
  return ct.get_tile_builder_bot_id(pos)

def find_core(ct: Controller):
  for x in ct.get_nearby_buildings():
    if ct.get_entity_type(x) == EntityType.CORE:
      return x
    
  return None

def get_delta_within_range(r) -> list[tuple[int, int]]:
  """
  Get all possible deltas (x, y) such that x^2 + y^2 <= range.
  """
  pairs = []
  limit = int(math.sqrt(r))

  for x in range(-limit, limit + 1):
    for y in range(-limit, limit + 1):
      if x * x + y * y <= r:
        pairs.append((x, y))

  return pairs

def neighbours_of_8(pos):
  x, y = pos
  return [
    (x + 1, y + 1), (x + 1, y - 1), (x - 1, y - 1), (x - 1, y + 1),
    (x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)
  ]

def neighbours_of_4(pos):
  x, y = pos
  return [
    (x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)
  ]

_DIRECTION_TABLE = {
  (0, -1): Direction.NORTH,
  (1, -1): Direction.NORTHEAST,
  (1, 0): Direction.EAST,
  (1, 1): Direction.SOUTHEAST,
  (0, 1): Direction.SOUTH,
  (-1, 1): Direction.SOUTHWEST,
  (-1, 0): Direction.WEST,
  (-1, -1): Direction.NORTHWEST,
  (0, 0): Direction.CENTRE,
}

def as_direction(dx, dy):
  return _DIRECTION_TABLE[(dx, dy)]
  
def l_inf(x, y):
  return max(abs(x[0] - x[1]), abs(y[0] - y[1]))

def l_1(x, y):
  return abs(x[0] - x[1]) + abs(y[0] - y[1])

def log(msg):
  print(msg, file=sys.stderr)

def sgn(x):
  return 1 if x > 0 else -1

def would_be_passable(ct: Controller, pos: Position):
  if not (ct.is_tile_empty(pos) or ct.is_tile_passable(pos)):
    return False
  
  id = get_id_at(ct, pos)
  return id is None or ct.get_entity_type(id) == EntityType.ROAD or ct.get_entity_type(id) == EntityType.CONVEYOR

class Player:
  def __init__(self):
    # =====
    # Fields for cores.
    # =====

    # Number of bots spawned in the begin phase.
    self.core_bots = 0
    # Current phase.
    self.phase = PHASE_BEGIN

    # =====
    # Fields for bots.
    # =====

    # The position of the core.
    self.core_pos = None
    # The type of this bot.
    self.state = BOT_UNASSIGNED
    # The target (destination) of this bot.
    self.tgt: Position = None
    # The map of the battlefield. Only concerns whether a grid is passable.
    self.passable_map: dict[tuple[int, int], bool] = {}
    # The next direction that the bot is about to take.
    self.next: tuple[int, int] = None
    # The number of turns that the bot is wandering.
    self.wandering = 1
    # Last direction in which the bot is wandering.
    self.last_dir = None
    # Whether the bot has been initialized after being put into conveyor state.
    self.conveyor_init = False
    # The ore deposite that the conveyor should originate from.
    self.ore_pos = None
    # The corner of the kernel where the conveyor is transmitted.
    self.transmitted_corner = None
    # The path of the conveyor.
    self.path = None
    # Current index of the convery path that we're processing.
    self.path_i = -1

    self.bot_visible = get_delta_within_range(GameConstants.BUILDER_BOT_VISION_RADIUS_SQ)

  def run(self, ct: Controller):
    etype = ct.get_entity_type()

    match etype:
      case EntityType.CORE:
        self.run_core(ct)
      case EntityType.BUILDER_BOT:
        self.run_bot(ct)
    
  def run_core(self, ct: Controller):
    id = ct.get_id()
    pos = ct.get_position(id)
    if self.phase == 0 and self.core_bots < PHASE_BEGIN_CORE_BOTS:
      spawn_pos = Position(pos.x - 1, pos.y - 1)
      if not ct.can_spawn(spawn_pos):
        return
      
      self.core_bots += 1
      ct.spawn_builder(spawn_pos)

  def run_bot(self, ct: Controller):
    x = time.perf_counter()
    id = ct.get_id()
    if self.state == BOT_UNASSIGNED:
      self.init_bot(ct)
    
    if self.state == BOT_EXPLORER:
      self.explore(ct, id)
    elif self.state == BOT_CONVEYOR_BUILDER:
      self.build_conveyor(ct, id)
    elif self.state == BOT_WAIT_FOR_HARVEST:
      self.build_harvester(ct, id)
    
    log(f"bot {id}: runtime {(time.perf_counter() - x) * 1000:.6f} ms")

  def init_bot(self, ct: Controller):
    core = find_core(ct)
    self.core_pos = ct.get_position(core)
    self.state = BOT_EXPLORER
    self.scan_map(ct)

  def find_target(self, ct: Controller, id: int):
    """
    Finds a target for the bot.
    """
    # Find a target.
    # Focus on getting ores.
    pos = ct.get_position(id)
    x, y = pos
    self.tgt = self.find_ore(ct)
    if self.tgt is not None:
      self.wandering = 0
      return
    
    # When we cannot find an ore deposit, we randomly choose a direction to explore.
    # We want it to be consistent (not moving north and then south in consecutive turns),
    # so we try last direction first.
    allowed = ct.get_nearby_tiles(2)
    if self.last_dir is not None:
      dx, dy = self.last_dir
      tgt = Position(x + dx, y + dy)
      if tgt in allowed and would_be_passable(ct, tgt):
        self.tgt = tgt
        self.wandering += 1
        return
      
      # When this hits a wall or another obstacle, we try reflecting the way.
      tgt = Position(x + dx, y - dy)
      if tgt in allowed and would_be_passable(ct, tgt):
        self.tgt = tgt
        self.last_dir = (dx, -dy)
        self.wandering += 1
        return
      tgt = Position(x - dx, y + dy)
      if tgt in allowed and would_be_passable(ct, tgt):
        self.tgt = tgt
        self.last_dir = (-dx, dy)
        self.wandering += 1
        return

    # When none of these work, we search all other directions.
    for w in range(id, id + 4):
      dx, dy = DIRECTIONS_DIAG[w & 3]
      tgt = Position(x + dx, y + dy)
      if tgt not in allowed:
        continue
      if would_be_passable(ct, tgt):
        self.tgt = tgt
        self.last_dir = (dx, dy)
        self.wandering += 1
        # We prefer empty tiles, so we continue searching when it's not empty.
        if ct.is_tile_empty(tgt):
          break

    if self.tgt is None:
      for w in range(id, id + 4):
        dx, dy = DIRECTIONS_STR[w & 3]
        tgt = Position(x + dx, y + dy)
        if tgt not in allowed:
          continue
        if would_be_passable(ct, tgt):
          self.tgt = tgt
          self.last_dir = (dx, dy)
          self.wandering += 1
          if ct.is_tile_empty(tgt):
            break
    pass #endif
    
  def scan_map(self, ct: Controller):
    for pos in ct.get_nearby_tiles():
      self.passable_map[(pos[0], pos[1])] = would_be_passable(ct, pos)

  def find_ore(self, ct: Controller):
    for pos in ct.get_nearby_tiles():
      env = ct.get_tile_env(pos)
      if (env == Environment.ORE_AXIONITE or env == Environment.ORE_TITANIUM) \
         and ct.get_tile_building_id(pos) == None and self.passable_map[(pos[0], pos[1])] == True:
        return pos
    
    return None
  
  def find_direction(self, ct: Controller):
    queue = []
    start = ct.get_position()
    goal = self.tgt
    closed = set()

    heapq.heappush(queue, (0, start))
    neighbours = neighbours_of_8 if self.state == BOT_EXPLORER else neighbours_of_4
    heuristic = l_inf if self.state == BOT_EXPLORER else l_1

    came_from = {}
    g = { start: 0 }

    while queue:
      _, current = heapq.heappop(queue)
      if current in closed:
        continue

      if current == goal:
        path = []
        while current in came_from:
          path.append(current)
          current = came_from[current]
        path.append(start)
        path.reverse()
        return path
      
      closed.add(current)

      for neighbour in neighbours(current):
        if not self.passable_map.get(neighbour, False):
          continue

        tentative = g[current] + 1

        if tentative < g.get(neighbour, float("inf")):
          came_from[neighbour] = current
          g[neighbour] = tentative
          f_score = tentative + heuristic(neighbour, goal)
          heapq.heappush(queue, (f_score, neighbour))

    return None
  
  def switch_to_explorer(self):
    self.state = BOT_EXPLORER
    self.tgt = None
    self.next = None
    self.wandering = 1
    self.last_dir = None

  def switch_to_conveyor_builder(self):
    self.path = None
    self.path_i = -1
    self.state = BOT_CONVEYOR_BUILDER
    self.conveyor_init = False

  def explore(self, ct: Controller, id: int):
    log(f"bot {id}: at: {ct.get_position()}")
    self.find_target(ct, id)
    if self.tgt is None:
      return

    log(f"bot {id}: target: {self.tgt}")

    # Do an A-star to obtain the next direction.
    if self.next is None:
      path = self.find_direction(ct)
      if path is None:
        log(f"warning: bot {id}: blocked by another bot!")
        self.tgt = None
        self.scan_map(ct)
        return
      
      self.next = path[1]
    pass # if end

    log(f"bot {id}: direction: {ct.get_position()} -> {self.next}")

    nx, ny = self.next
    x, y = ct.get_position()
    nextpos = Position(nx, ny)
    
    # The final step is not needed - we can build a harvester before we step on the tile.
    for dx, dy in DIRECTIONS:
      try:
        ore = Position(nx + dx, ny + dy)
        env = ct.get_tile_env(ore)
      except GameError:
        continue

      if not (env == Environment.ORE_AXIONITE or env == Environment.ORE_TITANIUM) or ct.get_tile_building_id(ore):
        continue

      self.ore_pos = ore
      cx, cy = self.core_pos
      self.transmitted_corner = Position(cx + sgn(x - cx), cy + sgn(y - cy))
      self.tgt = Position(cx + sgn(x - cx), cy + (sgn(y - cy) << 1))
      self.state = BOT_WAIT_FOR_HARVEST
      log(f"bot {id}: waiting to build a harvester")
      if ct.can_build_harvester(ore):
        self.build_harvester(ct, id)
        return

    if ct.can_build_road(nextpos):
      ct.build_road(nextpos)

    dir = as_direction(nx - x, ny - y)
    if not ct.can_move(dir):
      return
    
    ct.move(dir)
    # Scan the map after a move.
    self.scan_map(ct)

    # Clear the target info.
    if ct.get_position() == self.tgt:
      self.tgt = None
    self.next = None

  def build_harvester(self, ct: Controller, id: int):
    tileid = ct.get_tile_building_id(self.ore_pos)
    if tileid is not None:
      self.switch_to_explorer()
      return

    if not ct.can_build_harvester(self.ore_pos):
      return
    
    ct.build_harvester(self.ore_pos)
    log(f"bot {id}: built a harvester")
    self.scan_map(ct)
    self.switch_to_conveyor_builder()

  def build_conveyor(self, ct: Controller, id: int):
    if not self.conveyor_init:
      # We're now near the deposit (8-direction) but not necessary next to it (4-direction).
      # First move to one of direct position.
      pos = ct.get_position()
      ox, oy = self.ore_pos
      cx, cy = self.core_pos
      x, y = pos
      dx, dy = ox - x, oy - y
      if dx * dy == 0:
        self.conveyor_init = True
      else:
        mx, my = (dx, 0) if (abs(x - cx) + abs(oy - cy)) < (abs(ox - cx) + abs(y - cy)) else (0, dy)
        dst = Position(x + mx, y + my)
        dir = as_direction(mx, my)
        if ct.can_build_road(dst):
          ct.build_road(dst)
        if ct.can_move(dir):
          ct.move(dir)
          self.conveyor_init = True
        return
    pass #endif
    
    log(f"bot {id}: conveyor: {ct.get_position()}")
    if self.path is None:
      self.path = self.find_direction(ct)
      if self.path is None:
        log(f"warning: bot {id}: cannot find path!")
        self.scan_map(ct)
        return
      
      self.path.append(self.transmitted_corner)
      log(f"bot {id}: path is {self.path}")
    
    # When there are only two final nodes, the direction of the conveyor
    # is always pointing to the core tile.
    
    next, further = self.path[self.path_i + 1], self.path[self.path_i + 2]
    log(f"bot {id}: direction: {ct.get_position()} -> {next} -> {further}")
    nx, ny = next
    fx, fy = further
    pos = ct.get_position()
    x, y = pos
    nextpos = Position(nx, ny)

    # Destroy the road when there's one, to leave space for the conveyor.
    tileid = ct.get_tile_building_id(nextpos)
    ent = ct.get_entity_type(tileid) if tileid else None
    if ent == EntityType.ROAD and ct.can_destroy(nextpos):
      ct.destroy(nextpos)
    has_conveyor = ent == EntityType.CONVEYOR

    move_dir = as_direction(nx - x, ny - y)
    build_dir = as_direction(fx - nx, fy - ny)
    if not has_conveyor and ct.can_build_conveyor(nextpos, build_dir):
      ct.build_conveyor(nextpos, build_dir)
      has_conveyor = True

    # Special case for the first piece of conveyor.
    if self.path_i == -1 and has_conveyor:
      self.path_i = 0
      return

    # Only move after a conveyor has been built.
    if not has_conveyor or not ct.can_move(move_dir):
      return
    
    ct.move(move_dir)
    self.scan_map(ct)
    self.path_i += 1

    # Become an explorer again.
    if ct.get_position() == self.tgt:
      self.switch_to_explorer()
    