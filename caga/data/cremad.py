"""CREMA-D, 6-way emotion MCQ with a speaker-independent split.

Audio comes from the official GitHub repo CheyneyComputerScience/CREMA-D (AudioWAV/), with files
named {actor}_{sentence}_{EMO}_{intensity}.wav, e.g. 1001_DFA_ANG_XX.wav.

The split reproduces the TFDS `crema_d` speaker-independent split exactly (same algorithm and seed as
tensorflow_datasets/audio/crema_d.py) without installing TensorFlow:
    groups = sorted(set(speaker_id)); RandomState(0).shuffle(groups)
    probabilities train 0.7 / validation 0.1 / test 0.2 over the 91 speakers
The test split has 19 speakers (~1556 clips).
"""
import glob
import os
import subprocess

import numpy as np

from caga.data.common import dataset_dir, emotion_mcq, has_manifest, verify, write_manifest

NAME = "cremad"
ORDER = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EMOTION_CODE = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
                "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
SPLIT = "test"      # "test" | "validation" | "train" | "all" (7442 clips)
SPLIT_PROBS = [("train", 0.7), ("validation", 0.1), ("test", 0.2)]
REPO_URL = "https://github.com/CheyneyComputerScience/CREMA-D.git"


def _fetch_audio_wav(raw_dir):
    """Sparse-checkout only AudioWAV/ from the official repo and return its path."""
    repo = os.path.join(raw_dir, "CREMA-D")
    wav_dir = os.path.join(repo, "AudioWAV")
    if glob.glob(os.path.join(wav_dir, "*.wav")):
        return wav_dir
    os.makedirs(raw_dir, exist_ok=True)
    try:
        if not os.path.isdir(repo):
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                            REPO_URL, repo], check=True)
            subprocess.run(["git", "-C", repo, "sparse-checkout", "set", "AudioWAV"], check=True)
        else:
            subprocess.run(["git", "-C", repo, "sparse-checkout", "set", "AudioWAV"], check=False)
    except subprocess.CalledProcessError:
        if not os.path.isdir(repo):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, repo], check=True)
    return wav_dir


def _speaker_split(speaker_ids):
    """Map speaker id -> split name, identical to the TFDS crema_d split."""
    groups = sorted(set(speaker_ids))
    np.random.RandomState(0).shuffle(groups)
    n = len(groups)
    bounds, cum = [], 0.0
    for name, p in SPLIT_PROBS:
        prev = cum
        cum += p
        bounds.append((name, int(prev * n), int(cum * n)))
    bounds[-1] = (bounds[-1][0], bounds[-1][1], n)     # guard against rounding
    return {groups[i]: name for name, i0, i1 in bounds for i in range(i0, i1)}


def build():
    if has_manifest(NAME):
        verify(NAME)
        return
    wav_dir = _fetch_audio_wav(dataset_dir(NAME, "raw"))
    wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wavs:
        raise FileNotFoundError(f"No CREMA-D wavs under {wav_dir}; git clone may have failed.")

    parsed = []   # (path, id, speaker, emotion)
    for wav in wavs:
        base = os.path.basename(wav)[:-4]
        parts = base.split("_")
        if len(parts) < 3:
            continue
        emotion = EMOTION_CODE.get(parts[2])
        if emotion is not None:
            parsed.append((wav, base, parts[0], emotion))

    if SPLIT == "all":
        keep = parsed
    else:
        split_of = _speaker_split([p[2] for p in parsed])
        counts = {}
        for _, _, speaker, _ in parsed:
            counts[split_of[speaker]] = counts.get(split_of[speaker], 0) + 1
        print(f"  [{NAME}] TFDS-reproduced split clip counts: {counts}")
        keep = [p for p in parsed if split_of[p[2]] == SPLIT]

    rows = []
    for wav, base, _, emotion in keep:
        q, choices, gold = emotion_mcq(ORDER, emotion)
        rows.append(dict(id=f"{NAME}/{base}", dataset=NAME, audio_path=wav,
                         question=q, choices=choices, gold_idx=gold, domain="speech"))
    write_manifest(NAME, rows)
    verify(NAME)
    print(f"[{NAME}] {len(rows)} items (split = '{SPLIT}', 6-way emotion)")
