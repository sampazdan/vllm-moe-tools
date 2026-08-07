#!/opt/vllm-venv/bin/python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Create a prefix-based expert eligibility profile for MoE testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--keep", type=int, required=True)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layers <= 0:
        raise SystemExit("--layers must be positive")
    if args.experts <= 0:
        raise SystemExit("--experts must be positive")
    if not 0 < args.keep <= args.experts:
        raise SystemExit("--keep must be between 1 and --experts")
    if args.layer_start < 0:
        raise SystemExit("--layer-start must be nonnegative")

    keep = list(range(args.keep))
    profile = {
        "version": 1,
        "layers": {
            str(layer_id): {"keep": keep}
            for layer_id in range(args.layer_start, args.layer_start + args.layers)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
