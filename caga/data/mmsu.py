"""MMSU (5000 items) from Hugging Face ddwang2000/MMSU (split "train")."""
import os

from caga.data.common import (answer_idx_from, clean_choice, dataset_dir, decode_audio, get_answer,
                              get_audio_obj, get_choices, get_question, has_manifest, verify,
                              write_manifest)

NAME = "mmsu"


def build():
    if has_manifest(NAME):
        verify(NAME)
        return
    import soundfile as sf
    from datasets import load_dataset

    audio_dir = dataset_dir(NAME, "audio")
    ds = load_dataset("ddwang2000/MMSU", split="train")
    rows = []
    for i, ex in enumerate(ds):
        arr, sr = decode_audio(get_audio_obj(ex))
        if arr is None:
            if i == 0:
                print("WARN: could not decode audio; keys =", list(ex.keys()))
            continue
        wav = os.path.join(audio_dir, f"{i:05d}.wav")
        if not os.path.exists(wav):
            sf.write(wav, arr, sr)
        raw_choices = get_choices(ex)
        gold = answer_idx_from(get_answer(ex), raw_choices)
        if gold is None:
            continue
        rows.append(dict(id=f"{NAME}/{i:05d}", dataset=NAME, audio_path=wav,
                         question=get_question(ex), choices=[clean_choice(c) for c in raw_choices],
                         gold_idx=gold, domain=ex.get("domain", ex.get("task", ""))))
    write_manifest(NAME, rows)
    verify(NAME)
