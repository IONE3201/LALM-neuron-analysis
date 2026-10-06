"""RAVDESS speech (1440 clips, 24 actors) from Zenodo; 8-way emotion MCQ from the filename code."""
import glob
import os

from caga.data.common import (dataset_dir, download_file, emotion_mcq, extract, has_manifest, verify,
                              write_manifest)

NAME = "ravdess"
URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
EMOTION_CODE = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
ORDER = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def build():
    if has_manifest(NAME):
        verify(NAME)
        return
    audio_dir = dataset_dir(NAME, "audio")
    zip_path = os.path.join(dataset_dir(NAME, "raw"), "Audio_Speech_Actors_01-24.zip")
    download_file(URL, zip_path)
    if not glob.glob(os.path.join(audio_dir, "Actor_*")):
        extract(zip_path, audio_dir)
    rows = []
    for wav in sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)):
        emotion = EMOTION_CODE.get(os.path.basename(wav).split("-")[2])
        if emotion is None:
            continue
        q, choices, gold = emotion_mcq(ORDER, emotion)
        rows.append(dict(id=f"{NAME}/{os.path.basename(wav)[:-4]}", dataset=NAME, audio_path=wav,
                         question=q, choices=choices, gold_idx=gold, domain="speech"))
    write_manifest(NAME, rows)
    verify(NAME)
