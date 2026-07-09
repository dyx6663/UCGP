from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    if path_obj.suffix.lower() in {".json"}:
        return json.loads(text)
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for YAML configs. Install with `pip install pyyaml`.") from exc
    payload = yaml.safe_load(text)
    return payload or {}


def _expand_path(value: Any, base_dir: Path) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    value = os.path.expandvars(os.path.expanduser(value))
    if value.startswith("./") or value.startswith("../"):
        return str((base_dir / value).resolve())
    return value


def _normalize_paths(obj: Any, base_dir: Path) -> Any:
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key.lower().endswith(("dir", "path", "ckpt", "config", "model_id")):
                out[key] = _expand_path(value, base_dir)
            else:
                out[key] = _normalize_paths(value, base_dir)
        return out
    if isinstance(obj, list):
        return [_normalize_paths(item, base_dir) for item in obj]
    return obj


def _merge_run(defaults: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)
    merged.update(copy.deepcopy(run))
    if "output_root" in merged and "out_dir" not in merged and "OUT_DIR" not in merged:
        name = merged.get("name", "ucgp_run")
        merged["out_dir"] = str(Path(str(merged["output_root"])) / str(name))
    return merged


def load_experiment_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    cfg = _normalize_paths(_load_yaml_or_json(str(cfg_path)), cfg_path.parent)

    runtime = cfg.get("runtime", {}) or {}
    detector = cfg.get("detector", {}) or {}
    defaults = cfg.get("defaults", {}) or {}
    runs_raw = cfg.get("runs", []) or []
    if not runs_raw:
        raise ValueError("Config must contain at least one item under `runs`.")

    runs: List[Dict[str, Any]] = []
    for idx, run in enumerate(runs_raw):
        if not isinstance(run, dict):
            raise TypeError(f"runs[{idx}] must be a mapping.")
        runs.append(_merge_run(defaults, run))

    return {
        "runtime": runtime,
        "detector": detector,
        "runs": runs,
    }

