from __future__ import annotations

import argparse

from ucgp.engine import export_theta_pack_from_ckpt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the best UCGP theta from an optimization checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to de_ckpt_latest.npz.")
    parser.add_argument("--out-dir", required=True, help="Directory where the theta pack will be written.")
    parser.add_argument("--name", default="exported_uap_theta_pack.npz", help="Output .npz file name.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    export_theta_pack_from_ckpt(args.checkpoint, args.out_dir, export_name=args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

