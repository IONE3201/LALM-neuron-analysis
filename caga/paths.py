"""Storage locations, resolved at import time (no torch import, safe for CPU-only data preparation).

Every location can be overridden with an environment variable, set before `caga` is imported:

    CAGA_BASE     root of everything             (default: ~/caga)
    CAGA_DATA     manifests + audio              (default: $CAGA_BASE/datasets)
    CAGA_SCRATCH  large temporary files          (default: $CAGA_BASE/scratch)
    CAGA_RUNS     steering outputs               (default: $CAGA_BASE/caga_v30_2_runs)
    HF_HOME       Hugging Face cache (weights)   (default: $CAGA_BASE/models/hf)
"""
import os
import subprocess
import sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")   # must be set before huggingface_hub is imported

# Point CAGA_BASE at a disk with enough space (~60 GB for model weights, datasets and MELD extraction).
BASE = os.environ.get("CAGA_BASE", os.path.join(os.path.expanduser("~"), "caga"))
SCRATCH = os.environ.get("CAGA_SCRATCH", os.path.join(BASE, "scratch"))

HF_CACHE = os.path.join(BASE, "models", "hf")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

DATA_ROOT = os.environ.get("CAGA_DATA", os.path.join(BASE, "datasets"))
# The default directory name is kept from v30.2 so that existing runs can be resumed.
RUNS_BASE = os.environ.get("CAGA_RUNS", os.path.join(BASE, "caga_v30_2_runs"))

for _d in (BASE, SCRATCH, os.environ["HF_HOME"], DATA_ROOT, RUNS_BASE):
    os.makedirs(_d, exist_ok=True)


def bootstrap(ffmpeg=True):
    """Install the runtime dependencies needed for data preparation (best effort)."""
    def pip(*pkgs):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

    pip("huggingface_hub", "hf_transfer", "datasets", "soundfile", "librosa", "requests", "tqdm")
    import importlib.util
    if importlib.util.find_spec("hf_transfer") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    if ffmpeg:
        subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=False)


def show():
    print(f"BASE      = {BASE}")
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"HF_HOME   = {os.environ['HF_HOME']}")
    print(f"SCRATCH   = {SCRATCH}")
    print(f"RUNS_BASE = {RUNS_BASE}")
