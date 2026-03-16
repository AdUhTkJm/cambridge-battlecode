from cambc import Controller, Direction, EntityType, Environment, Position, GameConstants, GameError
from enum import Enum
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

# Bot types.
BOT_UNASSIGNED = 0
BOT_EXPLORER = 1

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

def neighbours_of(pos):
  x, y = pos
  return [
    (x + 1, y + 1), (x + 1, y - 1), (x - 1, y - 1), (x - 1, y + 1),
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
  
def a_star_heuristic(x, y):
  return max(abs(x[0] - x[1]), abs(y[0] - y[1]))

def log(msg):
  print(msg, file=sys.stderr)

def would_be_passable(ct: Controller, pos: Position):
  if not (ct.is_tile_empty(pos) or ct.is_tile_passable(pos)):
    return False
  
  id = get_id_at(ct, pos)
  if id is None or ct.get_entity_type(id) == EntityType.ROAD or ct.get_entity_type(id) == EntityType.CONVEYOR:
    return True

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
    self.bot_type = BOT_UNASSIGNED
    # The target (destination) of this bot.
    self.tgt: Position = None
    # The map of the battlefield. Only concerns whether a grid is passable.
    self.passable_map: dict[tuple[int, int], bool] = {}
    # The next direction that the bot is about to take.
    self.next: tuple[int, int] = None
    # The number of turns that the bot is wandering.
    self.wandering = 1
    # Width and height of the map.
    self.width = 256
    self.height = 256

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
    if self.bot_type == BOT_UNASSIGNED:
      self.init_bot(ct)
    
    if self.bot_type == BOT_EXPLORER:
      self.explore(ct, id)
    log(f"bot {id}: runtime {(time.perf_counter() - x) * 1000:.6f} ms")

  def init_bot(self, ct: Controller):
    core = find_core(ct)
    self.core_pos = ct.get_position(core)
    self.bot_type = BOT_EXPLORER
    self.scan_map(ct)

  def find_target(self, ct: Controller, id: int):
    """
    Finds a target for the bot.
    """
    # Find a target.
    # Focus on getting ores.
    pos = ct.get_position(id)
    self.tgt = self.find_ore(ct)
    if self.tgt is not None:
      log(f"bot {id}: guided by ore")
      self.wandering = 0
      return

    # When we cannot find an ore deposit, we randomly choose a direction to explore.
    # We want it to be consistent (not moving north and then south in consecutive turns),
    # so we're using `id` here: the direction will be the same for each bot.
    start = id + (self.wandering >> 3)
    allowed = ct.get_nearby_tiles(2)
    for w in range(start, start + 4):
      dx, dy = DIRECTIONS_DIAG[w & 3]
      tgt = Position(pos.x + dx, pos.y + dy)
      if tgt not in allowed:
        continue
      if would_be_passable(ct, tgt):
        self.tgt = tgt
        # self.next = (tgt[0], tgt[1])
        self.wandering += 1
        # We prefer empty tiles, so we continue searching when it's not empty.
        if ct.is_tile_empty(tgt):
          break

    if self.tgt is None:
      for w in range(start, start + 4):
        dx, dy = DIRECTIONS_STR[w & 3]
        tgt = Position(pos.x + dx, pos.y + dy)
        if tgt not in allowed:
          continue
        if would_be_passable(ct, tgt):
          self.tgt = tgt
          # self.next = (tgt[0], tgt[1])
          self.wandering += 1
          if ct.is_tile_empty(tgt):
            break
    pass # if end
    
  def scan_map(self, ct: Controller):
    self.passable_map = {}
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

      for neighbour in neighbours_of(current):
        if not self.passable_map.get(neighbour, False):
          continue

        tentative = g[current] + 1

        if tentative < g.get(neighbour, float("inf")):
          came_from[neighbour] = current
          g[neighbour] = tentative
          f_score = tentative + a_star_heuristic(neighbour, goal)
          heapq.heappush(queue, (f_score, neighbour))

    return None

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
    for d in Direction:
      ore = ct.get_position().add(d)
      if ct.can_build_harvester(ore):
        log(f"bot {id}: built a harvester")
        ct.build_harvester(ore)
        self.tgt = None
        self.next = None
        self.scan_map(ct)
        return

    if ct.can_build_road(nextpos):
      ct.build_road(nextpos)

    dir = as_direction(nx - x, ny - y)
    if not ct.can_move(dir):
      # Maybe something changed that blocked the movement.
      # Try again.
      return
    
    ct.move(dir)
    # Scan the map after a move.
    self.scan_map(ct)

    # Clear the target info.
    if ct.get_position() == self.tgt:
      self.tgt = None
    self.next = None
