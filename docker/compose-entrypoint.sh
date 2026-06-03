#!/usr/bin/env bash
set -euo pipefail

action="${1:-run}"
shift || true

dataset="${CONRAG_DATASET:-example}"

case "${action}" in
  run)
    exec python -u main.py --dataset "${dataset}" --mode run "$@"
    ;;
  build)
    exec python -u main.py --dataset "${dataset}" --mode build "$@"
    ;;
  query)
    exec python -u main.py --dataset "${dataset}" --mode query "$@"
    ;;
  debug-web)
    exec python -u -m conrag.debug_web "$@"
    ;;
  populate-fanout)
    exec python -u populate_fanout.py "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${action}" "$@"
    ;;
esac
