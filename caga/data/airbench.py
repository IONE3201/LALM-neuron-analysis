"""AIR-Bench Foundation (24,683 single-audio MCQs, ~27 GB) from qyang1021/AIR-Bench-Dataset."""
import glob
import json
import os
from collections import Counter

from caga.data.common import answer_idx_from, clean_choice, dataset_dir, has_manifest, verify, write_manifest

NAME = "airbench_foundation"
MAX_PER_TASK = None    # e.g. 100 keeps at most 100 items per task (~1,900 items); None keeps all


def build():
    if has_manifest(NAME):
        verify(NAME)
        return
    from huggingface_hub import snapshot_download

    raw = dataset_dir(NAME, "raw")
    snap = snapshot_download("qyang1021/AIR-Bench-Dataset", repo_type="dataset",
                             local_dir=raw, allow_patterns=["Foundation/**"])
    fdir = os.path.join(snap, "Foundation")
    metas = glob.glob(os.path.join(fdir, "**", "Foundation_meta.json"), recursive=True) or \
        glob.glob(os.path.join(fdir, "**", "*.json"), recursive=True)
    print("foundation metas:", [os.path.relpath(m, snap) for m in metas])
    found = {os.path.basename(p): p for ext in ("wav", "flac", "mp3")
             for p in glob.glob(os.path.join(fdir, "**", f"*.{ext}"), recursive=True)}

    rows, kept = [], Counter()
    for meta_path in metas:
        with open(meta_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", list(data.values())[0])
        print(os.path.basename(meta_path), "keys:", list(data[0].keys()), "| n:", len(data))
        for ex in data:
            task = ex.get("task_name", "")
            if MAX_PER_TASK and kept[task] >= MAX_PER_TASK:
                continue
            raw_choices = [ex.get(f"choice_{l}") for l in "abcd" if ex.get(f"choice_{l}") is not None]
            if not raw_choices:
                continue
            gold = answer_idx_from(ex.get("answer_gt", ex.get("answer")), raw_choices)
            local = found.get(os.path.basename(str(ex.get("path", ""))))
            if gold is None or not local:
                continue
            rows.append(dict(id=f"{NAME}/{ex.get('uniq_id')}", dataset=NAME, audio_path=local,
                             question=ex.get("question", ""), choices=[clean_choice(c) for c in raw_choices],
                             gold_idx=gold, domain=task))
            kept[task] += 1
    write_manifest(NAME, rows)
    verify(NAME)
    print(f"[{NAME}] {len(rows)} items (per-task cap = {MAX_PER_TASK})")
