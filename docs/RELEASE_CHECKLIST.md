# Release Checklist

Before pushing this repository to GitHub:

- Replace all placeholder paths in `configs/*.yaml`.
- Confirm that private datasets, checkpoints, logs, and generated outputs are not tracked.
- Run `python -m ucgp.cli --config configs/smoke_test.yaml --dry-run`.
- Run the smoke test on a small local image subset.
- Render one exported theta pack with `tools/render_theta.py`.
- Update the BibTeX entry after the paper metadata is final.
- Confirm whether `LICENSE` should remain MIT or be changed to your lab/project policy.

Useful checks:

```bash
rg "password|token|api_key|secret|PRIVATE|Bearer|autodl|/root|D:\\\\"
git status --short
```

