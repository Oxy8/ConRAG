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
  shell)
    exec bash "$@"
    ;;
  *)
    exec "${action}" "$@"
    ;;
esac
