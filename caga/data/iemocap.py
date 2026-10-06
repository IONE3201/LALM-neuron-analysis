"""IEMOCAP, standard 4-class setting (angry / happy+excited / sad / neutral).

Uses the public mirror AbstractTTS/IEMOCAP. The canonical 5531-utterance set is approximated with a
"unique top vote (no tie)" rule on the mirror's soft labels, which gives ~5528 utterances. The exact
5531 set requires the licensed USC EmoEvaluation labels.
"""
import os

import numpy as np

from caga.data.common import (EMOTION_QUESTION, dataset_dir, decode_audio, has_manifest, verify,
                              write_manifest)

NAME = "iemocap"
VOTE_COLUMNS = ["frustrated", "angry", "sad", "disgust", "excited", "fear", "neutral", "surprise", "happy"]
TARGET = {"angry", "happy", "excited", "sad", "neutral"}
MERGE = {"excited": "happy"}
ORDER = ["angry", "happy", "sad", "neutral"]


def build():
    if has_manifest(NAME):
        verify(NAME)
        return
    import soundfile as sf
    from datasets import load_dataset

    audio_dir = dataset_dir(NAME, "audio")
    ds = load_dataset("AbstractTTS/IEMOCAP", split="train")
    rows = []
    for i, ex in enumerate(ds):
        votes = np.array([float(ex[k]) for k in VOTE_COLUMNS])
        order = votes.argsort()[::-1]
        top = VOTE_COLUMNS[order[0]]
        if top not in TARGET:
            continue
        if not votes[order[0]] > votes[order[1]] + 1e-9:     # majority agreement: unique top vote
            continue
        emotion = MERGE.get(top, top)
        arr, sr = decode_audio(ex["audio"])
        if arr is None:
            continue
        fn = os.path.basename(ex.get("file", f"{i:05d}.wav"))
        wav = os.path.join(audio_dir, fn)
        if not os.path.exists(wav):
            sf.write(wav, arr, sr)
        rows.append(dict(id=f"{NAME}/{fn[:-4]}", dataset=NAME, audio_path=wav,
                         question=EMOTION_QUESTION, choices=list(ORDER),
                         gold_idx=ORDER.index(emotion), domain="speech"))
    write_manifest(NAME, rows)
    verify(NAME)
    print(f"[{NAME}] {len(rows)} items (standard 4-class; canonical = 5531)")
