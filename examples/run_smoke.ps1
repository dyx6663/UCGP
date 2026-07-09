$ErrorActionPreference = "Stop"

python -m ucgp.cli `
  --config configs/smoke_test.yaml `
  --output-json outputs_smoke/summary.json

