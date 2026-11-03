#!/usr/bin/env python3
"""Fetch the pinned WSP-Bench base model into the host-only HF cache."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEFAULT_CACHE = Path("/srv/wsp-cache/hf")


def fetch(cache_dir: Path, endpoint: str | None = None) -> Path:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(cache_dir)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mirror", action="store_true", help="use hf-mirror.com")
    parser.add_argument(
        "--fallback-after",
        type=float,
        default=0,
        metavar="SECONDS",
        help="retry through hf-mirror.com if the first attempt fails after this many seconds",
    )
    args = parser.parse_args()

    started = time.monotonic()
    try:
        path = fetch(args.cache_dir, "https://hf-mirror.com" if args.mirror else None)
    except Exception:
        elapsed = time.monotonic() - started
        if args.mirror or not args.fallback_after or elapsed < args.fallback_after:
            raise
        path = fetch(args.cache_dir, "https://hf-mirror.com")
    print(path)


if __name__ == "__main__":
    main()
