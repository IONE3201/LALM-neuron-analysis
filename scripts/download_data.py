#!/usr/bin/env python3
"""Download benchmarks and write unified manifests (CPU only).

    python scripts/download_data.py                      # default set (all except AIR-Bench)
    python scripts/download_data.py mmau_mini mmar       # selected datasets
    python scripts/download_data.py airbench_foundation  # ~27 GB
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caga import paths  # noqa: E402
from caga.data import DEFAULT, REGISTRY, build, data_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"datasets to build, from {list(REGISTRY)}")
    ap.add_argument("--bootstrap", action="store_true", help="pip-install data dependencies and ffmpeg first")
    args = ap.parse_args()
    if args.bootstrap:
        paths.bootstrap()
    paths.show()
    names = args.datasets or DEFAULT
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        ap.error(f"unknown dataset(s) {unknown}; valid: {list(REGISTRY)}")
    build(*names)
    data_summary(names)


if __name__ == "__main__":
    main()
