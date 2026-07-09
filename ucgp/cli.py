from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ucgp",
        description="Optimize Universal Curved-Grid Patch (UCGP) for infrared VLM auditing.",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved config without running.")
    parser.add_argument("--device", default=None, help="Override runtime device, e.g. cuda:0 or cpu.")
    parser.add_argument("--output-json", default=None, help="Optional path for a machine-readable run summary.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_experiment_config(args.config)
    runtime = dict(cfg.get("runtime", {}))
    detector = dict(cfg.get("detector", {}))
    runs = list(cfg.get("runs", []))

    if args.device:
        runtime["device"] = args.device

    if args.dry_run:
        print(json.dumps({"runtime": runtime, "detector": detector, "runs": runs}, ensure_ascii=False, indent=2))
        return 0

    from . import engine

    engine.configure_runtime(
        seed=runtime.get("seed"),
        device=runtime.get("device"),
        log_level=runtime.get("log_level"),
        clip_batch_size=runtime.get("clip_batch_size"),
        save_images_on_best_full=runtime.get("save_images_on_best_full"),
    )

    results = engine.run_many(
        runs=runs,
        detector_config=detector.get("config"),
        detector_ckpt=detector.get("checkpoint"),
        score_thr=detector.get("score_thr", 0.5),
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
