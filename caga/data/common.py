"""Shared helpers for building benchmark manifests.

Every benchmark is converted to one `manifest.jsonl` under DATA_ROOT/<name>/ with the schema

    id · dataset · audio_path · question · choices · gold_idx · domain
"""
import json
import os
import re
import tarfile
import zipfile

import numpy as np

from caga.paths import DATA_ROOT

EMOTION_QUESTION = "What is the emotion expressed in this audio?"


# ---------------------------------------------------------------- manifest I/O
def dataset_dir(name, *sub):
    p = os.path.join(DATA_ROOT, name, *sub)
    os.makedirs(p, exist_ok=True)
    return p


def manifest_file(name):
    return os.path.join(DATA_ROOT, name, "manifest.jsonl")


def has_manifest(name):
    p = manifest_file(name)
    return os.path.exists(p) and os.path.getsize(p) > 0


def write_manifest(name, rows):
    p = manifest_file(name)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{name}] wrote {len(rows)} rows -> {p}")


def read_manifest(name):
    with open(manifest_file(name)) as f:
        return [json.loads(line) for line in f if line.strip()]


def verify(name, n_show=1):
    rows = read_manifest(name)
    missing = sum(not os.path.exists(r["audio_path"]) for r in rows)
    print(f"[{name}] {len(rows)} items | missing audio: {missing}")
    for r in rows[:n_show]:
        print("   e.g.", {k: r[k] for k in ("id", "domain", "choices", "gold_idx")})


# ---------------------------------------------------------------- download / extract
def download_file(url, dest, chunk=1 << 20):
    import requests
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("  [skip] exists:", os.path.basename(dest))
        return dest
    print("  [get]", url)
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for c in r.iter_content(chunk):
                if c:
                    f.write(c)
    os.replace(tmp, dest)
    return dest


def extract(archive, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if archive.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(out_dir)
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    print("  [extracted]", os.path.basename(archive), "->", out_dir)


# ---------------------------------------------------------------- MCQ helpers
def emotion_mcq(emotion_order, gold_emotion):
    """Fixed-order emotion MCQ -> (question, choices, gold_idx)."""
    choices = list(emotion_order)
    return EMOTION_QUESTION, choices, choices.index(gold_emotion)


def clean_choice(c):
    """Strip a leading option label such as "(A) ", "A) ", "A. " or "A: "."""
    c = str(c).strip()
    c = re.sub(r"^\(?[A-Za-z]\)\s*", "", c)
    c = re.sub(r"^[A-Za-z][\.\:]\s*", "", c)
    return c.strip()


def answer_idx_from(ans, raw_choices):
    """Map an answer (option label or option text) to a 0-based choice index, or None."""
    s = str(ans).strip()
    m = re.match(r"\(([A-Za-z])\)|([A-Za-z])[\.\):]", s)    # (A) / A) / A. / A:  (not a bare "A bird")
    if m:
        i = ord((m.group(1) or m.group(2)).upper()) - 65
        if 0 <= i < len(raw_choices):
            return i
    a = clean_choice(s).lower()
    for i, c in enumerate(raw_choices):
        if clean_choice(c).lower() == a:
            return i
    return None


# ---------------------------------------------------------------- Hugging Face schema detection
def get_question(ex):
    return ex.get("instruction") or ex.get("question") or ex.get("prompt") or ""


def get_choices(ex):
    return ex.get("choices") or ex.get("options") or \
        [ex.get(f"choice_{l}") for l in "abcd" if ex.get(f"choice_{l}")]


def get_answer(ex):
    v = ex.get("answer")
    return v if v is not None else ex.get("answer_gt", ex.get("label"))


def get_audio_obj(ex):
    def is_audio(v):
        return hasattr(v, "get_all_samples") or (isinstance(v, dict) and "array" in v)

    for k in ("context", "audio", "wav", "speech"):
        v = ex.get(k)
        if v is not None and is_audio(v):
            return v
    for v in ex.values():
        if hasattr(v, "get_all_samples") or (isinstance(v, dict) and "array" in v and "sampling_rate" in v):
            return v
    return None


def decode_audio(obj):
    """Decode a `datasets` audio object (dict with "array", or a torchcodec AudioDecoder) to mono."""
    if isinstance(obj, dict) and "array" in obj:
        return np.asarray(obj["array"], dtype="float32"), int(obj["sampling_rate"])
    if hasattr(obj, "get_all_samples"):                 # datasets>=4 decodes via torchcodec
        s = obj.get_all_samples()
        d = s.data
        arr = (d.mean(0) if d.ndim == 2 else d).cpu().numpy().astype("float32")
        return arr, int(s.sample_rate)
    return None, None
