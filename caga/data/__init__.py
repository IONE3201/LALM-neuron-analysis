"""Benchmark registry: every module exposes build(), which writes DATA_ROOT/<name>/manifest.jsonl."""
import os

from caga.data import airbench, cremad, iemocap, meld, mmar, mmau, mmsu, ravdess
from caga.data.common import has_manifest, read_manifest

REGISTRY = {
    "mmau_mini": mmau,
    "mmar": mmar,
    "mmsu": mmsu,
    "ravdess": ravdess,
    "iemocap": iemocap,
    "meld": meld,
    "cremad": cremad,
    "airbench_foundation": airbench,
}

# Built by default. AIR-Bench (~27 GB) must be requested explicitly.
DEFAULT = ["mmau_mini", "mmar", "mmsu", "ravdess", "iemocap", "meld", "cremad"]


def build(*names):
    """build("mmau_mini", "mmar", ...), or build() for every dataset in DEFAULT."""
    names = names or DEFAULT
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown dataset(s) {unknown}; valid: {list(REGISTRY)}")
    for n in names:
        print(f"\n########## building {n} ##########")
        REGISTRY[n].build()


def data_summary(names=None):
    """Print a coverage table (items and audio files present) for the given datasets."""
    names = names or list(REGISTRY)
    print(f"{'dataset':<22}{'items':>8}{'audio_ok':>10}   status")
    print("-" * 54)
    for n in names:
        if has_manifest(n):
            rows = read_manifest(n)
            ok = sum(os.path.exists(r["audio_path"]) for r in rows)
            print(f"{n:<22}{len(rows):>8}{ok:>10}   OK")
        else:
            print(f"{n:<22}{'-':>8}{'-':>10}   MISSING (run build('{n}'))")
    print("\nschema: id · dataset · audio_path · question · choices · gold_idx · domain")
