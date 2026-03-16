#!/bin/zsh

die() {
  echo -e "\e[31merror:\e[0m $1"
}
warn() {
  echo -e "\e[33mwarning:\e[0m $1"
}

while [[ $# -ne 0 ]]; do
  case "$1" in
  -t) cambc run starter defect 2> output.txt; shift;;
  -r) cambc watch replay.replay26 2> /dev/null; shift;;
  *) die "unknown option: $1"; exit 1;;
  esac
done
