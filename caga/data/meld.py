"""MELD dev + test (3718 utterances), 7-way emotion MCQ. Audio is extracted from the mp4 clips.

The raw archive (~10 GB, ~20 GB extracted) is unpacked under SCRATCH; only the 16 kHz mono wavs are
kept under DATA_ROOT.
"""
import csv
import glob
import os
import subprocess
import tarfile

from caga.data.common import EMOTION_QUESTION, dataset_dir, has_manifest, verify, write_manifest
from caga.paths import SCRATCH

NAME = "meld"
ORDER = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
SPLITS = {"dev": ("dev_sent_emo.csv", "dev_splits_complete"),
          "test": ("test_sent_emo.csv", "output_repeated_splits_test")}   # add "train" for more data


def _untar(archive, out):
    with tarfile.open(archive) as t:
        t.extractall(out)


def build():
    if has_manifest(NAME):
        verify(NAME)
        return
    from huggingface_hub import hf_hub_download

    audio_dir = dataset_dir(NAME, "audio")
    local = os.path.join(SCRATCH, "meld_raw")
    os.makedirs(local, exist_ok=True)
    tar = hf_hub_download("declare-lab/MELD", "MELD.Raw.tar.gz", repo_type="dataset", local_dir=local)
    print("downloaded ->", tar)
    base = os.path.join(local, "MELD.Raw")
    if not os.path.exists(base):
        _untar(tar, local)
    for inner in glob.glob(os.path.join(base, "*.tar.gz")):
        _untar(inner, base)

    rows = []
    for split, (csv_name, video_dir_name) in SPLITS.items():
        csv_paths = glob.glob(os.path.join(base, "**", csv_name), recursive=True)
        video_dirs = glob.glob(os.path.join(base, "**", video_dir_name), recursive=True)
        if not csv_paths or not video_dirs:
            print("  [warn] missing", split, csv_name)
            continue
        with open(csv_paths[0], errors="ignore") as f:
            for r in csv.DictReader(f):
                d, u = r["Dialogue_ID"], r["Utterance_ID"]
                emotion = r["Emotion"].strip().lower()
                if emotion not in ORDER:
                    continue
                mp4 = os.path.join(video_dirs[0], f"dia{d}_utt{u}.mp4")
                if not os.path.exists(mp4):
                    continue
                wav = os.path.join(audio_dir, f"{split}_dia{d}_utt{u}.wav")
                if not os.path.exists(wav):
                    subprocess.run(["ffmpeg", "-y", "-i", mp4, "-ac", "1", "-ar", "16000", wav],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if not os.path.exists(wav):
                    continue
                rows.append(dict(id=f"{NAME}/{split}_d{d}_u{u}", dataset=NAME, audio_path=wav,
                                 question=EMOTION_QUESTION, choices=list(ORDER),
                                 gold_idx=ORDER.index(emotion), domain="speech"))
    write_manifest(NAME, rows)
    verify(NAME)
