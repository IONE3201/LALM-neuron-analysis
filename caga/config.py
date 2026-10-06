"""All experiment settings in one place.

Settings are plain module attributes and are read at call time, so they can be changed from a
notebook or script before calling `caga.experiment.run_steering`:

    from caga import config
    config.SELECT_SCORE = "aad"
    config.MODES = ["text"]
"""
import os

# ---------------------------------------------------------------- model
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"          # only the Thinker (text-output) part is loaded
DEVICE = "cuda"
DTYPE = "bfloat16"                          # torch dtype name
SYSTEM_PROMPT = "You are a helpful audio assistant."
MAX_NEW_TOKENS = 16                         # MCQ answers are a single letter; no chain of thought
AUDIO_TOKEN_ID = 151646                     # <|AUDIO|> placeholder id in the Qwen2.5-Omni tokenizer
SAMPLE_RATE = 16000

# ---------------------------------------------------------------- contrast pair
# A = clean audio, B = Gaussian noise with the same length and RMS (SNR 0 dB).
CONTRAST_SEED = 2024
CONTRAST_SNR_DB = 0.0
# What the per-item noise seed is derived from. "path" (v30.2 behaviour) makes the noise depend on where
# the dataset is stored, so it differs between machines; "id" is portable but changes the noise.
CONTRAST_SEED_KEY = "path"

# ---------------------------------------------------------------- attribution
# Score used to rank neurons (see REPORT.md, section 2.4):
#   contrast      salA - salB, ranked by absolute value          (proposed method)
#   clean_abs     salA,        ranked by absolute value
#   clean_signed  salA,        top positive
#   noise_abs     salB,        ranked by absolute value
#   noise_signed  salB,        top positive
#   aad           actA - actB, top positive (no gradient; activation-difference baseline)
SELECT_SCORE = "contrast"
SELECT_SCORES = ["contrast", "clean_abs", "clean_signed", "noise_abs", "noise_signed", "aad"]
ESTIMATOR = "gradxact"                      # gradient x activation (= integrated gradients with 1 step)
IG_STEPS = 1

# ---------------------------------------------------------------- steering
# Where neurons are selected / injected (see REPORT.md, section 2.5).
MODES = ["final", "audio", "text", "audio_text", "per_token"]
ALL_MODES = ["final", "all", "audio", "text", "audio_text", "per_token"]
K_LIST = [10000]                            # number of (layer, neuron) pairs per position set
ALPHAS = [-1.0, -0.3, 0.0, 0.3, 1.0]        # injection strength; 0.0 is the unsteered baseline
NORM = "none"                               # "none" or "l2" (unit-norm the injected vector)
# False: one shared top-K neuron set per position set (salience summed over positions).
# True:  every position picks its own top-K (affects audio / text / all / audio_text only).
PERTOKEN_TOPK = False

# ---------------------------------------------------------------- datasets
DATASETS = ["mmau_mini", "mmar", "mmsu", "ravdess", "iemocap", "meld", "airbench_foundation"]

# ---------------------------------------------------------------- performance
CPU_THREADS = int(os.environ.get("CAGA_THREADS", "8"))   # avoid oversubscription on shared hosts
BATCH_ALPHAS = True       # generate all alphas of one (mode, K) in a single batched call
FAST_ATTR = False         # keep dAB on GPU; tiny bf16 differences vs CPU, so off by default
FREE_CACHE = False        # gc + empty_cache after each item; enable only on CUDA OOM

# ---------------------------------------------------------------- output layout
# Kept identical to v30.2 so that earlier runs are found and resumed.
RESULTS_SUBDIR = "caga_eval_v30_2"
