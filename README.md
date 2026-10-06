# CAGA: Per-Item Neuron Attribution and Activation Steering for Audio LLMs

CAGA finds, **for each test item separately**, the MLP neurons of an audio-language model that carry audio-driven information, and steers them at inference time. It uses no labels and no training.

For every item, the model runs once on the clean audio (A) and once on loudness-matched Gaussian noise (B). Neurons are scored with gradient × activation of the KL divergence between the two runs' answer distributions. The top-K neurons are then shifted by `alpha * (act_A - act_B)` during the prefill pass. The current target is **Qwen2.5-Omni-7B (Thinker)**, evaluated as single-audio multiple choice; data builders exist for 8 benchmarks.

The full method description, experiments, results and limitations are in **[REPORT.md](REPORT.md)**.

```
clean audio A ─┐                                ┌─ salA = actA · ∂KL(pA‖pB)/∂actA ─┐
               ├─ Qwen2.5-Omni Thinker (MLP act) ┤                                   ├─ score ─ top-K ─┐
noise  audio B ─┘   (same prompt, same length)   └─ salB = actB · ∂KL(pB‖pA)/∂actB ─┘                  │
                                                                                                        ▼
                         generation with  act[l][pos, neurons] += alpha · (actA − actB)[l][pos, neurons]
```

---

## Repository layout

```
caga/                       Python package
  config.py                 all experiment settings (model, contrast, selection, modes, K, alphas, speed)
  paths.py                  storage locations (CAGA_BASE / CAGA_DATA / CAGA_SCRATCH / CAGA_RUNS)
  data/                     benchmark builders -> DATA_ROOT/<name>/manifest.jsonl
    common.py               manifest I/O, download, audio decoding, option parsing
    mmau.py mmar.py mmsu.py ravdess.py iemocap.py meld.py cremad.py airbench.py
  mcq.py                    prompt formatting, gold letter, answer parsing
  model.py                  model loading, input construction, greedy generation
  contrast.py               clean / Gaussian-noise contrast pair
  attribution.py            two-sided grad x act on the KL objective, selection scores
  selection.py              top-K neuron selection per mode -> steering specs
  steering.py               forward hooks that inject alpha * delta during prefill
  experiment.py             run_steering(), evaluate()
  runs.py                   result file layout and readers (no torch)
  results.py                result tables, Excel export, paired McNemar test (no torch)
  selftest.py               GPU self-tests for the speed options
scripts/
  download_data.py          build manifests from the command line
  run_steering.py           run attribution + steering from the command line
  summarize_results.py      tables / Excel / CSV from finished runs
notebooks/
  01_prepare_data.ipynb     download benchmarks
  02_run_steering.ipynb     run experiments (GPU)
  03_analyze_results.ipynb  tables and significance tests
results/                    numbers extracted from the experiments run so far (CSV)
legacy/                     original v30.2 code and notebooks with their outputs (kept for reference)
REPORT.md                   method and experiment report
requirements.txt
setup_env.sh
```

---

## Installation

**Linux GPU machine.** Set `CAGA_BASE` to a disk with about 60 GB free, then run:

```bash
export CAGA_BASE=/path/to/large/disk/caga
bash setup_env.sh             # ffmpeg, venv on $CAGA_BASE, torch (cu124), requirements, Jupyter kernel "Python (caga)"
```

Edit the torch index URL in `setup_env.sh` to match `nvidia-smi` (cu118 / cu121 / cu124 …).

**Manual install.**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
sudo apt-get install -y ffmpeg
```

**Hardware.** All experiments in this repository were run on a single **NVIDIA A100 80GB** GPU. Qwen2.5-Omni-7B in bf16 needs about 16 GB for the weights, and attribution needs extra memory for activations and gradients, so a GPU with less than 24 GB is not enough. Data preparation and result analysis run on CPU.

---

## Storage

Every location is under `CAGA_BASE` and can be overridden with an environment variable:

`CAGA_BASE` defaults to `~/caga`.

| what | default | override |
|---|---|---|
| manifests + audio | `$CAGA_BASE/datasets` | `CAGA_DATA` |
| Hugging Face cache (model weights, ~16 GB) | `$CAGA_BASE/models/hf` | `HF_HOME` |
| large temporary files (MELD extraction, ~20 GB) | `$CAGA_BASE/scratch` | `CAGA_SCRATCH` |
| predictions and tables | `$CAGA_BASE/caga_v30_2_runs` | `CAGA_RUNS` |

Make sure the Hugging Face cache and the scratch directory are on a disk with enough space; a small system disk will run out during model download or MELD extraction.

---

## Quick start

```bash
# 1. data (CPU)
python scripts/download_data.py mmau_mini mmar

# 2. sanity run on a small benchmark, then the real one (GPU)
python scripts/run_steering.py mmau_mini --modes final text --eval
python scripts/run_steering.py mmar --eval --excel

# 3. tables from finished runs (CPU, no torch)
python scripts/summarize_results.py mmar --csv results/mmar_runs.csv
```

The same steps from Python:

```python
from caga import config
from caga.data import build
from caga.experiment import run_steering, evaluate
from caga.results import collect, paired_test

build("mmar")

config.SELECT_SCORE = "contrast"          # or clean_abs / clean_signed / noise_abs / noise_signed / aad
config.MODES = ["final", "audio", "text", "audio_text", "per_token"]
config.K_LIST = [10000]
config.ALPHAS = [-1.0, -0.3, 0.0, 0.3, 1.0]
run_steering(["mmar"])
evaluate(["mmar"])

df = collect("mmar")                      # every finished configuration of this dataset
paired_test("mmar", "final", 10000, -1.0) # exact McNemar test vs alpha = 0 on the same items
```

Runs are **resumable**. Results go to one file per (dataset, select score, mode, K, alpha), and items already in a file are skipped. Different select scores and modes write to different directories, so they never overwrite each other.

---

## Main options (`caga/config.py`)

| option | default | meaning |
|---|---|---|
| `SELECT_SCORE` | `"contrast"` | neuron ranking score: `contrast` (salA − salB, by absolute value), `clean_abs`, `clean_signed`, `noise_abs`, `noise_signed`, or `aad` (actA − actB, no gradient) |
| `MODES` | 5 modes | `final` (last position), `audio` (audio-token positions), `text` (all other positions), `audio_text` (both, 2K budget), `per_token` (each audio position picks its own top-K); `all` is also available |
| `K_LIST` | `[10000]` | number of (layer, neuron) pairs steered (out of 28 × 18,944) |
| `ALPHAS` | `[-1, -0.3, 0, 0.3, 1]` | injection strength; −1 = replace by the noise run, 0 = baseline, +1 = amplify the difference |
| `PERTOKEN_TOPK` | `False` | per-position top-K for `audio` / `text` / `all` / `audio_text` (directories get a `_pt` suffix) |
| `NORM` | `"none"` | `"l2"` normalizes the injected vector |
| `CONTRAST_SNR_DB` | `0.0` | noise loudness relative to the clean audio |
| `CONTRAST_SEED_KEY` | `"path"` | seed the noise by audio path (v30.2 behavior) or by item `"id"` (portable across machines) |
| `BATCH_ALPHAS` | `True` | generate all alphas of a configuration in one batch (verified identical) |
| `FAST_ATTR` | `False` | keep dAB on GPU (tiny bf16 differences; off for consistency) |
| `FREE_CACHE` | `False` | per-item `empty_cache()`; enable only on CUDA OOM |
| `CAGA_THREADS` (env) | `8` | CPU thread cap; prevents oversubscription on shared hosts |

`scripts/run_steering.py --help` lists the matching command-line flags.

---

## Benchmarks

| name | items | notes |
|---|---|---|
| `mmau_mini` | 1000 | MMAU test-mini (HF `AudioLLMs/MMAU-mini`) |
| `mmar` | 1000 | audio from HF `BoJack/MMAR`, gold answers from the official GitHub metadata |
| `mmsu` | 5000 | HF `ddwang2000/MMSU` |
| `ravdess` | 1440 | speech, 8 emotions |
| `iemocap` | ~5528 | 4-class standard set from a public mirror (canonical 5531 needs licensed labels) |
| `meld` | 3718 | dev + test, 7 emotions, audio extracted from mp4 |
| `cremad` | ~1556 | TFDS-reproduced speaker-independent test split, 6 emotions |
| `airbench_foundation` | 24,683 | ~27 GB; `caga.data.airbench.MAX_PER_TASK` keeps a subset |

The manifest schema is `id · dataset · audio_path · question · choices · gold_idx · domain`. To add a benchmark, write a module in `caga/data/` that exposes a `build()` function writing this schema, and register it in `caga/data/__init__.py`.

---

## Output format

```
$CAGA_RUNS/caga_eval_v30_2/<dataset>/<estimator>[_<select>]_<mode>_<norm>_K<K>[_pt]/alpha_<tag>.jsonl
    {"id": "mmar/...", "raw": "a", "pred": "a", "gold": "a", "correct": true}
```

`<select>` is omitted for `contrast`. `alpha_<tag>` encodes alpha as `m1`, `m0p3`, `0`, `0p3`, `1`. This layout is the same as v30.2, so earlier runs are found and resumed (point `CAGA_RUNS` at the existing result directory).

---

## Self-tests

Run these once per new environment (GPU):

```python
from caga import selftest
selftest.selftest_alpha_batch("mmau_mini")   # batched alphas must match per-alpha generation
selftest.selftest_fast_attr("mmau_mini")     # only before enabling FAST_ATTR
selftest.time_one_item("mmar")               # attribution vs generation time
```

---

## Migrating from the v30.2 notebooks

| v30.2 (`caga.partb`) | now |
|---|---|
| `partb.load_model()` | `caga.model.load_model()` (also called by `run_steering`) |
| `partb.run_v30_2(ds)` / `partb.eval_v30_2(ds)` | `caga.experiment.run_steering(ds)` / `evaluate(ds)` |
| `partb.SELECT_SCORE`, `V30_MODES`, `V30_K_LIST`, `V30_ALPHAS`, `V30_NORM` | `config.SELECT_SCORE`, `MODES`, `K_LIST`, `ALPHAS`, `NORM` |
| `partb.PERTOKEN_TOPK`, `BATCH_ALPHAS`, `FAST_ATTR`, `FREE_CACHE` | same names in `caga.config` |
| `compute_scope_attribution(wav, noise, prompt, m)` | `caga.attribution.compute_attribution(clean, noise, prompt)` |
| `build_specs(attr, mode, K, norm)` | `caga.selection.build_specs(attr, mode, K)` |
| `PerItemRunner(processor, model, specs, T)` | `caga.steering.Steerer(specs, T)` |
| `caga.helpers`, `caga.adapter` | `caga.data.common`, `caga.mcq` |
| aggregation cells copied between notebooks | `caga.results.collect / save_excel / paired_test` |

On a toy model, the refactored code gives bit-identical attribution, neuron selection and predictions to `partb.py` for every selection score and mode (REPORT.md, Section 4.3).
