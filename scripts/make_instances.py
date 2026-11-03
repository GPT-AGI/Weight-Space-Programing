#!/usr/bin/env python3
"""Write public dev specs and private ground-truth recipes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsp.gym.instances import build_instance_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--test-config",
        type=Path,
        help="private JSON containing test seeds, triggers, and prompt words",
    )
    args = parser.parse_args()
    instances = build_instance_set(args.split, test_config=args.test_config)
    for instance in instances:
        print(instance.id)


if __name__ == "__main__":
    main()
