#!/usr/bin/env bash
set -euo pipefail

python -m ucgp.cli \
  --config configs/paper_default.yaml \
  --output-json outputs/paper_default_summary.json

