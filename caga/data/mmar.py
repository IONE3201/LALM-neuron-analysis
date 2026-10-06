"""MMAR (1000 items). Audio from Hugging Face BoJack/MMAR, metadata from the official GitHub jsonl.

The GitHub metadata stores the answer as the verbatim text of one choice, so the gold index is exact.
"""
import glob
import json
import os
import urllib.request

from caga.data.common import dataset_dir, extract, has_manifest, verify, write_manifest
from caga.paths import DATA_ROOT

NAME = "mmar"
META_URL = "https://raw.githubusercontent.com/ddlBoJack/MMAR/main/MMAR-meta.jsonl"


def _find_wavs():
    pattern = os.path.join(DATA_ROOT, NAME, "**", "*.wav")
    return {os.path.basename(p): p for p in glob.glob(pattern, recursive=True)}


def build():
    if has_manifest(NAME):
        verify(NAME)
        return
    raw = dataset_dir(NAME, "raw")
    found = _find_wavs()
    if not found:
        from huggingface_hub import snapshot_download
        snapshot_download("BoJack/MMAR", repo_type="dataset", local_dir=raw, allow_patterns=["*.tar.gz"])
        for archive in glob.glob(os.path.join(raw, "**", "*.tar.gz"), recursive=True):
            extract(archive, os.path.dirname(archive))
        found = _find_wavs()

    meta_path = os.path.join(raw, "MMAR-meta.jsonl")
    urllib.request.urlretrieve(META_URL, meta_path)
    with open(meta_path) as f:
        meta = [json.loads(line) for line in f if line.strip()]
    rows = []
    for ex in meta:
        local = found.get(os.path.basename(ex.get("audio_path", "")))
        if not local:
            continue
        choices = ex.get("choices") or []
        ans = str(ex.get("answer", "")).strip().lower()
        gold = next((i for i, c in enumerate(choices) if str(c).strip().lower() == ans), None)
        if gold is None:
            continue
        rows.append(dict(id=f"{NAME}/{ex.get('id')}", dataset=NAME, audio_path=local,
                         question=ex.get("question", ""), choices=list(choices), gold_idx=gold,
                         domain=ex.get("modality", "")))
    write_manifest(NAME, rows)
    verify(NAME)
    print(f"[{NAME}] matched {len(rows)}/{len(meta)} (official GitHub metadata, exact gold)")
