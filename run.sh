#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run.py --rows "${ROWS:-120000}" "$@"
