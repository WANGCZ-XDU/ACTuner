#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python3}"

usage() {
  echo "Usage: bash run_server.sh --target-recall {0.90|0.92|0.94|0.96|0.98|0.99}"
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 2 || "$1" != "--target-recall" ]]; then
  usage >&2
  exit 2
fi
case "$2" in
  0.9|0.90|0.92|0.94|0.96|0.98|0.99) ;;
  *) usage >&2; exit 2 ;;
esac
export TARGET_RECALL="$2"
export PYTHONDONTWRITEBYTECODE=1
TARGET_TAG="${TARGET_RECALL/./}0"
TARGET_TAG="${TARGET_TAG:0:3}"
RUN_ID="${RUN_ID:-actuner_R${TARGET_TAG}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-runs/$RUN_ID}"
if [[ -e "$LOG_DIR" ]]; then
  echo "Result directory already exists: $LOG_DIR" >&2
  exit 3
fi

"$PYTHON" -B tests/test_server_preflight.py
mkdir -p "$LOG_DIR"
"$PYTHON" -B -u tuner.py --log-dir "$LOG_DIR" --target-recall "$TARGET_RECALL" \
  2>&1 | tee "$LOG_DIR/console.log"
