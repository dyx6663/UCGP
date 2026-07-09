from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ucgp.engine import MASTER_SIZE, generate_master_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an exported UCGP theta pack as a mask image.")
    parser.add_argument("--theta-pack", required=True, help="Path to exported_uap_theta_pack.npz.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--size", type=int, default=None, help="Override render size.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pack = np.load(args.theta_pack, allow_pickle=True)
    theta = np.asarray(pack["theta"], dtype=np.float32)
    grid = int(pack["G"]) if "G" in pack.files else 5
    size = int(args.size or (int(pack["MASTER_SIZE"]) if "MASTER_SIZE" in pack.files else MASTER_SIZE))
    mask = generate_master_mask(theta, G=grid, size=size)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
    print(f"Rendered theta pack to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

