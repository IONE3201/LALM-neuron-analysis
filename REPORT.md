# CAGA: Per-Item Contrastive Attribution and Activation Steering of MLP Neurons in an Audio LLM

**Experiment report, code version v30.2 (refactored October 2026)**

---

## 0. Summary

**Question.** When an audio LLM answers a multiple-choice question about an audio clip, which MLP neurons carry the information that comes from the audio? Can we find them for each item separately, without labels, and change the answer by steering only those neurons?

**Method (CAGA).** For every test item, run the model twice with the same prompt: once with the clean audio (A) and once with Gaussian noise of the same length and loudness (B). Score each MLP neuron (layer, unit) with gradient × activation of the KL divergence between the two runs' next-token distributions, computed on both sides. Combine the two sides into a selection score and keep the top-K neurons. During the prefill pass, add `alpha * (act_A - act_B)` to those neurons and decode greedily. The method uses no gold labels and no training data: it adapts to each item at test time.

**Setup.** Qwen2.5-Omni-7B (Thinker) on a single NVIDIA A100 80GB, single-audio MCQ, 6 benchmarks (MMAU-mini, MMAR, MMSU, RAVDESS, IEMOCAP, MELD), alpha ∈ {−1, −0.3, 0, +0.3, +1}, K from 500 to 50,000 neurons (out of 28 × 18,944 ≈ 530k), five position modes, and six selection scores.

**Main findings so far.**

1. **The selected neurons matter causally.** Pushing them toward the noise run (alpha = −1) at audio and text positions lowers accuracy steadily as K grows, on every dataset. For example, `audio_text` mode on RAVDESS drops from 0.634 to 0.116 at K = 50k, and on MMAR from 0.623 to 0.361 (Table 3).
2. **The only consistent gain comes from `final` mode with negative alpha.** Steering only the last prompt position *toward* the noise run raises accuracy on MMAR (+2.4 to +3.3 points), IEMOCAP (+1.2 to +1.6), and RAVDESS (+0.9 to +2.0). The gain is almost the same for every K from 500 to 50k. It does not appear on MMAU-mini, MMSU, or MELD (Tables 2 and 4).
3. **Other modes give at most about ±1 to 2 points.** Results depend on the dataset, and the best alpha is usually +0.3 on MMAR and −0.3 on the emotion datasets.
4. **The selection score matters a lot for disruption but little for gains.** `clean_signed` (top positive salience on the clean side) is by far the most destructive set. On IEMOCAP `audio` mode, accuracy falls to 0.23–0.29 at alpha < 0 and to 0.63 at alpha = +0.3. For `final` mode, the gradient-free baseline `aad` (activation difference only) is as good as or better than the gradient-based scores: MMAR +3.0, IEMOCAP +1.7 (Table 5).
5. **Caveats (Section 6).** The best alpha is chosen on the test set itself, so the reported gains are optimistic. Most gains are about the size of the noise between software environments (the unsteered baseline differs by up to 0.6 points between the two experiment runs). No paired significance test or random-neuron control has been run yet. The proposed `contrast` score was never run with the KL loss on the datasets where gains appeared, so the comparison of selection scores is still incomplete.

---

## 1. Motivation

Large audio-language models (LALMs) often answer audio questions from the text prompt and language priors instead of from the audio. Neuron-level analysis can show where the audio evidence enters the language model. If those neurons can be found reliably, they can be used for:

* **interpretation**: how much of the decision depends on audio-driven neurons, and at which positions (audio tokens, text tokens, or the final answer position);
* **intervention**: whether strengthening or weakening those neurons changes the answer in a predictable way.

Neuron-finding methods usually aggregate over a dataset (for example, neurons that respond to a concept across many examples). CAGA is **per item**: the neuron set is recomputed for every test example from that example's own clean-versus-noise contrast. No label is needed, so it can in principle be used at test time.

---

## 2. Method

### 2.1 Model and the unit of analysis

* **Model:** `Qwen/Qwen2.5-Omni-7B`, Thinker part only (audio encoder + Qwen2.5-7B language model, text output), bf16.
* **Neuron:** one coordinate of the MLP activation in a decoder layer. Each layer computes `down_proj( act_fn(gate_proj(x)) * up_proj(x) )`, and a neuron is one entry of the `act_fn(gate_proj(x))` output, which is read and written through a forward hook on `mlp.act_fn`. The model has L = 28 layers with D = 18,944 neurons each, about 530k neurons in total. A neuron is defined per token position, so the activation of layer l is a `[T, D]` matrix for a prompt of T tokens.
* **Prompt:** the system prompt "You are a helpful audio assistant.", then the user turn with the audio, the question, the lettered options `(a) … (b) …`, and "Answer with only the letter." Decoding is greedy with at most 16 new tokens. The first letter a–z found in the output is the prediction (`caga/mcq.py`).

### 2.2 Contrast pair

For each item:

* **A (clean):** the original audio, resampled to 16 kHz mono.
* **B (noise):** white Gaussian noise with the **same length** as A and the **same RMS** (SNR 0 dB), clipped to [−1, 1]. The seed comes from `crc32(audio path) XOR 2024`.

A and B have the same length, so the audio encoder produces the same number of audio tokens for both. The two prompts therefore line up position by position, and the activation difference `dAB[l] = act_A[l] − act_B[l]` (shape `[T, D]`) is well defined at every position.

### 2.3 Attribution: two-sided gradient × activation on a KL objective

Let `p_A` and `p_B` be the next-token distributions at the **last prompt position**, which is the position that produces the answer letter. For each side, we use the other side's distribution as a fixed target and take gradient × activation:

```
salA[l, t, i] = act_A[l, t, i] · ∂ KL(p_A ‖ p_B) / ∂ act_A[l, t, i]        (target p_B detached)
salB[l, t, i] = act_B[l, t, i] · ∂ KL(p_B ‖ p_A) / ∂ act_B[l, t, i]        (target p_A detached)
```

Intuitively, `salA` measures how much each clean-run neuron pushes the clean prediction away from the noise prediction, and `salB` does the same in the other direction.

Implementation details (`caga/attribution.py`):

* This is integrated gradients with one step (`IG_STEPS = 1`), which is plain gradient × activation. Multi-step IG existed in v30.1 and was removed in v30.2 for speed.
* All 28 `act_fn` outputs are replaced by leaf tensors in a single forward pass, so the gradient of each layer is read from its own leaf. As a result, gradient paths that go through **later layers' gate activations** are blocked; paths through attention, `up_proj`, and the residual stream are kept. Each layer's salience is therefore a partial derivative that holds later gate activations fixed.
* Salience is aggregated per **scope**: `final` (last position only), `audio` (sum over the `<|AUDIO|>` placeholder positions), `text` (sum over all other positions, including the system prompt and chat-template tokens), and `all`. If needed, the full per-token tensor `[L, T, D]` is also kept.
* The cost per item is 2 forward passes without gradients + 2 forward/backward passes. With the `aad` score, no backward pass is needed.
* Earlier versions used a total-variation (TV) loss on the same distributions. The current code uses KL (the function was historically still named `_tv_loss`; it is renamed `kl_last_token` here).

### 2.4 Selection scores

The ranking score decides which K (layer, neuron) pairs are steered. Six variants are implemented (`config.SELECT_SCORE`):

| name | score | ranking | role |
|---|---|---|---|
| `contrast` | salA − salB | top-K by absolute value | **proposed method** |
| `clean_abs` | salA | top-K by absolute value | clean side only |
| `clean_signed` | salA | top-K positive | clean side, sign-aware |
| `noise_abs` | salB | top-K by absolute value | noise side only |
| `noise_signed` | salB | top-K positive | noise side, sign-aware |
| `aad` | act_A − act_B | top-K positive | gradient-free baseline ("activation-difference") |

For the shared modes below, the score is first summed over the positions of the scope and then ranked, so signed contributions from different positions can cancel out.

### 2.5 Where to steer: modes

The **injection vector is always the per-token activation difference `dAB`**, restricted to the selected neurons and positions. The mode decides which positions are used and which salience scope ranks the neurons:

| mode | neurons ranked by | injected at | budget |
|---|---|---|---|
| `final` | salience at the last position | last prompt position only | K |
| `audio` | salience summed over audio positions | every audio position (one shared neuron set) | K |
| `text` | salience summed over text positions | every text position (one shared neuron set) | K |
| `audio_text` | `audio` and `text` ranked separately | both sets at their own positions | 2K |
| `per_token` | each **audio** position's own salience | each audio position gets its own top-K | K per position |
| `all` | salience summed over all positions | every position | K (implemented, not used in experiments) |

`config.PERTOKEN_TOPK = True` switches `audio`, `text`, `all`, and `audio_text` to per-position selection, like `per_token`. Results then go to directories with a `_pt` suffix (`scripts/run_steering.py --per-token-topk`). This switch is implemented, but no complete results with it are recorded in the notebooks.

### 2.6 Steering

During the **prefill** pass only (sequence length > 1), a forward hook on `mlp.act_fn` of each affected layer does:

```
act[l][:, positions, neurons] += alpha · dAB[l][positions, neurons]
```

Decoding steps are not modified, but they attend to the modified prefill keys and values, and the first answer token is produced directly by the modified last position. On the selected coordinates, alpha has a simple meaning:

* **alpha = −1**: the clean activation becomes the noise-run activation, so the audio-specific signal on those neurons is removed (a patching or ablation toward noise).
* **alpha = 0**: no change. This is the unsteered baseline and is produced by the same code path.
* **alpha = +1**: the activation becomes `2·act_A − act_B`, so the clean-versus-noise difference is amplified.
* **alpha = ±0.3**: smaller steps in either direction.

The vector is not normalized (`NORM = "none"`). An L2-normalized variant exists but was not used.

### 2.7 Evaluation protocol

* Every (dataset, select score, mode, K, alpha, item) produces one JSON line `{id, raw, pred, gold, correct}`. The metric is accuracy.
* "best Δ" in the tables is the best **non-zero** alpha minus alpha = 0, where the best alpha is **chosen on the evaluation set** (an oracle choice; see Section 6).
* Runs can be resumed at item level.

### 2.8 Compute

* **Hardware:** all experiments ran on a single NVIDIA A100 80GB GPU.
* About 7.3 s per MMAR item (1000 items in about 2 h) for 5 modes × 5 alphas at one K, including both attribution sides. The `aad` score skips the backward passes.
* For one MMAR item with 24 configurations, attribution took 2.3 s versus 12.4 s for generation, so attribution is about 15% of the time. Generation dominates the cost.
* Speed options (all verified, Section 4.3): build the model inputs once per item, run all alphas of one configuration as a single batch of 5 (`BATCH_ALPHAS`), rank neurons once per item and reuse the ranking for every K, cap CPU threads (`CAGA_THREADS`), and skip per-item `empty_cache`.

---

## 3. Benchmarks

All benchmarks are converted to one schema (`id, dataset, audio_path, question, choices, gold_idx, domain`) by `caga/data/*.py`.

| dataset | items | source and construction | answer format |
|---|---|---|---|
| MMAU-mini | 1000 | HF `AudioLLMs/MMAU-mini` (test-mini) | 4-way MCQ, letter or text answer matched to options |
| MMAR | 1000 | audio: HF `BoJack/MMAR`; metadata: official GitHub `MMAR-meta.jsonl` (answer is the exact option text) | MCQ |
| MMSU | 5000 | HF `ddwang2000/MMSU`, split `train` | MCQ, text match |
| RAVDESS | 1440 | Zenodo speech set, 24 actors | 8-way emotion from the filename code |
| IEMOCAP | 5528 | HF mirror `AbstractTTS/IEMOCAP`; standard 4 classes (angry, happy + excited, sad, neutral) with a unique top vote | 4-way emotion (canonical set: 5531; the exact set needs the licensed labels) |
| MELD | 3718 | HF `declare-lab/MELD`, dev + test, mp4 → 16 kHz wav | 7-way emotion |
| CREMA-D | ~1556 | official GitHub, TFDS-reproduced speaker-independent test split | 6-way emotion (**added, not yet run**) |
| AIR-Bench Foundation | 24,683 | HF `qyang1021/AIR-Bench-Dataset` | MCQ (**builder ready, not yet run**) |

Emotion datasets use a fixed question ("What is the emotion expressed in this audio?") and a fixed option order.

---

## 4. Experiments and results

All numbers below come from the original notebook outputs. They are machine-extracted into `results/exp1_k_sweep_contrast.csv` and `results/exp2_select_score_ablation_kl.csv`, and the original notebooks are kept, translated, in `legacy/`. Accuracies are fractions and Δ values are percentage points.

### 4.1 Experiment 1: K sweep with the `contrast` score

* **Settings:** `SELECT_SCORE = contrast`; modes `final / audio / text / audio_text`; K ∈ {500, 1k, 5k, 10k, 20k, 50k}; 5 alphas; 6 datasets. Source: `legacy/exp1_k_sweep_contrast.ipynb`. Hardware: NVIDIA A100 80GB.
* **Loss:** this run predates the KL-titled notebook, and the v30.2 README at the time described the objective as a **TV loss**. The exact loss used is not recorded in the outputs, so treat it as "TV, probably". Re-running with the current KL code is advisable before comparing with Experiment 2.
* **Completeness:** MMAU-mini, MMAR, MMSU, and RAVDESS are complete. MELD is **partial (1456 / 3718)**. IEMOCAP files contain **5537–5538 lines for 5528 items**: a few items were written twice, probably by an interrupted or concurrent run. The old evaluator counted lines, and the new one (`caga/runs.py`) counts unique ids.

**Table 2: best Δ over the alpha = 0 baseline at K = 10k (best alpha in parentheses).** \* = incomplete (see above).

| Dataset | n | α=0 | final | audio | text | audio_text |
|---|---|---|---|---|---|---|
| MMAU-mini | 1000 | 0.698 | +0.0 (−0.3) | −0.8 (+1.0) | −0.6 (+0.3) | −0.6 (+0.3) |
| MMAR | 1000 | 0.623 | **+2.8 (−1.0)** | +1.9 (+0.3) | +1.3 (+1.0) | +1.1 (+0.3) |
| MMSU | 5000 | 0.650 | +0.0 (+1.0) | −0.1 (+0.3) | −0.1 (+0.3) | +0.1 (+0.3) |
| RAVDESS | 1440 | 0.634 | +1.0 (−1.0) | +0.3 (−0.3) | +1.1 (−0.3) | +1.3 (−0.3) |
| IEMOCAP | 5528\* | 0.693 | +1.2 (−1.0) | +0.2 (+0.3) | +1.0 (−0.3) | −0.8 (+0.3) |
| MELD | 1456 / 3718\* | 0.501 | +0.5 (−0.3) | +0.3 (−0.3) | +1.2 (−0.3) | +1.0 (−0.3) |

**Table 3: accuracy at alpha = −1 (selected neurons replaced by noise-run values) as a function of K, `audio_text` mode.**

| Dataset | α=0 | K=500 | 1k | 5k | 10k | 20k | 50k |
|---|---|---|---|---|---|---|---|
| MMAU-mini | 0.698 | 0.675 | 0.660 | 0.629 | 0.597 | 0.555 | 0.499 |
| MMAR | 0.623 | 0.555 | 0.529 | 0.489 | 0.458 | 0.416 | 0.361 |
| MMSU | 0.650 | 0.583 | 0.564 | 0.493 | 0.447 | 0.388 | 0.327 |
| RAVDESS | 0.634 | 0.517 | 0.471 | 0.319 | 0.235 | 0.174 | 0.116 |
| IEMOCAP\* | 0.693 | 0.562 | 0.528 | 0.400 | 0.354 | 0.311 | 0.247 |
| MELD\* | 0.501 | 0.382 | 0.330 | 0.236 | 0.193 | 0.172 | 0.145 |

**Table 4: accuracy at alpha = −1 as a function of K, `final` mode.**

| Dataset | α=0 | K=500 | 1k | 5k | 10k | 20k | 50k |
|---|---|---|---|---|---|---|---|
| MMAU-mini | 0.698 | 0.699 | 0.698 | 0.694 | 0.696 | 0.699 | 0.695 |
| MMAR | 0.623 | **0.656** | 0.648 | 0.651 | 0.651 | 0.650 | 0.647 |
| MMSU | 0.650 | 0.646 | 0.644 | 0.644 | 0.644 | 0.645 | 0.643 |
| RAVDESS | 0.634 | **0.654** | 0.646 | 0.646 | 0.644 | 0.644 | 0.643 |
| IEMOCAP\* | 0.693 | 0.708 | **0.709** | **0.709** | 0.706 | 0.706 | 0.705 |
| MELD\* | 0.501 | 0.489 | 0.494 | 0.489 | 0.490 | 0.488 | 0.487 |

Observations:

* **Monotone damage with K (Table 3).** At audio and text positions, replacing more selected neurons with their noise-run values steadily removes the information needed for the task. The emotion datasets are the most sensitive: RAVDESS falls below the 8-way chance level of 0.125 at K = 50k. This is evidence that the attribution finds neurons that carry audio-driven information. A random-neuron control with the same K is still needed to show that the *selection* is better than chance (Section 7).
* **Positive alpha does not help at audio and text positions.** At alpha = +1, accuracy is usually at or below the baseline (for example, RAVDESS `audio` at K = 50k: 0.573). Amplifying the clean-versus-noise difference is not a reliable way to improve answers.
* **`final` mode behaves differently (Table 4).** On MMAR, RAVDESS, and IEMOCAP, *subtracting* the audio-induced shift at the last position improves accuracy, and the gain hardly depends on K. On the same datasets, alpha = +1 lowers accuracy (RAVDESS 0.60–0.61, IEMOCAP 0.66–0.67), so accuracy decreases as alpha increases. One possible reading is that part of the last-position shift caused by the audio is a bias, for example toward certain emotion labels, rather than evidence. This is a hypothesis and has not been tested. Because the gain is flat in K, a small number of neurons, or the largest-magnitude ones, may account for most of the effect.
* **MMAU-mini and MMSU are insensitive.** No setting improves them by more than 0.3 points.

### 4.2 Experiment 2: selection-score ablation with the KL loss

* **Settings:** KL loss, K = 10k, 5 modes (including `per_token`), 5 alphas. Source: `legacy/exp2_select_score_ablation_kl.ipynb`. Hardware: NVIDIA A100 80GB.
* **What was actually run:**

| dataset | select scores completed | notes |
|---|---|---|
| MMAR | clean_signed, aad, clean_abs, noise_signed | `contrast` and `noise_abs` were **not** run with KL |
| MMSU | contrast | `clean_abs` run interrupted |
| IEMOCAP | clean_abs, clean_signed, noise_abs, aad | `contrast` and `noise_signed` not run with KL |
| MELD, MMAU-mini | — | cells prepared, never executed |

**Table 5: MMAR (n = 1000, KL, K = 10k).**

| select | mode | α=−1 | α=−0.3 | α=0 | α=+0.3 | α=+1 | best Δ (α) |
|---|---|---|---|---|---|---|---|
| clean_signed | final | 0.627 | 0.639 | 0.629 | 0.626 | 0.632 | +1.0 (−0.3) |
| clean_signed | audio | 0.335 | 0.393 | 0.629 | 0.599 | 0.582 | −3.0 (+0.3) |
| clean_signed | text | 0.380 | 0.538 | 0.629 | 0.623 | 0.601 | −0.6 (+0.3) |
| clean_signed | audio_text | 0.291 | 0.354 | 0.629 | 0.592 | 0.578 | −3.7 (+0.3) |
| clean_signed | per_token | 0.263 | 0.278 | 0.629 | 0.579 | 0.561 | −5.0 (+0.3) |
| clean_abs | final | 0.652 | 0.641 | 0.629 | 0.626 | 0.630 | +2.3 (−1.0) |
| clean_abs | audio | 0.556 | 0.618 | 0.629 | 0.639 | 0.625 | +1.0 (+0.3) |
| clean_abs | text | 0.567 | 0.632 | 0.629 | 0.639 | 0.630 | +1.0 (+0.3) |
| clean_abs | audio_text | 0.426 | 0.614 | 0.629 | 0.630 | 0.630 | +0.1 (+0.3) |
| clean_abs | per_token | 0.529 | 0.622 | 0.629 | 0.634 | 0.621 | +0.5 (+0.3) |
| noise_signed | final | 0.640 | 0.643 | 0.629 | 0.631 | 0.618 | +1.4 (−0.3) |
| noise_signed | audio | 0.619 | 0.629 | 0.629 | 0.640 | 0.630 | +1.1 (+0.3) |
| noise_signed | text | 0.359 | 0.625 | 0.629 | 0.637 | 0.614 | +0.8 (+0.3) |
| noise_signed | audio_text | 0.316 | 0.610 | 0.629 | 0.633 | 0.623 | +0.4 (+0.3) |
| noise_signed | per_token | 0.548 | 0.638 | 0.629 | 0.628 | 0.632 | +0.9 (−0.3) |
| aad | final | **0.659** | 0.636 | 0.629 | 0.626 | 0.621 | **+3.0 (−1.0)** |
| aad | audio | 0.590 | 0.631 | 0.629 | 0.632 | 0.621 | +0.3 (+0.3) |
| aad | text | 0.599 | 0.635 | 0.629 | **0.642** | 0.632 | +1.3 (+0.3) |
| aad | audio_text | 0.530 | 0.627 | 0.629 | 0.639 | 0.628 | +1.0 (+0.3) |
| aad | per_token | 0.520 | 0.626 | 0.629 | 0.632 | 0.622 | +0.3 (+0.3) |

**Table 6: IEMOCAP (n = 5528, KL, K = 10k).**

| select | mode | α=−1 | α=−0.3 | α=0 | α=+0.3 | α=+1 | best Δ (α) |
|---|---|---|---|---|---|---|---|
| clean_abs | final | 0.707 | 0.703 | 0.693 | 0.684 | 0.663 | +1.4 (−1.0) |
| clean_abs | audio | 0.530 | 0.676 | 0.693 | 0.695 | 0.694 | +0.3 (+0.3) |
| clean_abs | text | 0.619 | 0.700 | 0.693 | 0.682 | 0.658 | +0.7 (−0.3) |
| clean_abs | audio_text | 0.336 | 0.671 | 0.693 | 0.684 | 0.663 | −0.8 (+0.3) |
| clean_abs | per_token | 0.544 | 0.675 | 0.693 | 0.693 | 0.693 | +0.1 (+1.0) |
| clean_signed | final | 0.667 | 0.698 | 0.693 | 0.683 | 0.665 | +0.5 (−0.3) |
| clean_signed | audio | 0.231 | 0.285 | 0.693 | 0.629 | 0.628 | −6.4 (+0.3) |
| clean_signed | text | 0.360 | 0.635 | 0.693 | 0.655 | 0.629 | −3.7 (+0.3) |
| clean_signed | audio_text | 0.224 | 0.255 | 0.693 | 0.628 | 0.627 | −6.5 (+0.3) |
| clean_signed | per_token | 0.226 | 0.227 | 0.693 | 0.627 | 0.628 | −6.4 (+1.0) |
| noise_abs | final | 0.708 | 0.700 | 0.693 | 0.686 | 0.666 | +1.5 (−1.0) |
| noise_abs | audio | 0.653 | 0.690 | 0.693 | 0.696 | 0.692 | +0.3 (+0.3) |
| noise_abs | text | 0.644 | 0.701 | 0.693 | 0.682 | 0.662 | +0.9 (−0.3) |
| noise_abs | audio_text | 0.534 | 0.692 | 0.693 | 0.685 | 0.664 | −0.1 (−0.3) |
| noise_abs | per_token | 0.667 | 0.690 | 0.693 | 0.693 | 0.686 | −0.0 (+0.3) |
| aad | final | **0.710** | 0.700 | 0.693 | 0.686 | 0.671 | **+1.7 (−1.0)** |
| aad | audio | 0.597 | 0.679 | 0.693 | 0.698 | **0.702** | +0.9 (+1.0) |
| aad | text | 0.697 | 0.696 | 0.693 | 0.687 | 0.676 | +0.5 (−1.0) |
| aad | audio_text | 0.514 | 0.681 | 0.693 | 0.692 | 0.689 | −0.1 (+0.3) |
| aad | per_token | 0.564 | 0.672 | 0.693 | 0.696 | 0.697 | +0.4 (+1.0) |

**Table 7: MMSU (n = 5000, KL, K = 10k, `contrast`).**

| mode | α=−1 | α=−0.3 | α=0 | α=+0.3 | α=+1 | best Δ (α) |
|---|---|---|---|---|---|---|
| final | 0.643 | 0.649 | 0.651 | 0.648 | 0.650 | −0.0 (+1.0) |
| audio | 0.580 | 0.644 | 0.651 | 0.650 | 0.643 | −0.0 (+0.3) |
| text | 0.588 | 0.641 | 0.651 | 0.648 | 0.647 | −0.2 (+0.3) |
| audio_text | 0.446 | 0.633 | 0.651 | 0.647 | 0.648 | −0.2 (+1.0) |
| per_token | 0.545 | 0.646 | 0.651 | 0.647 | 0.638 | −0.3 (+0.3) |

Observations:

* **`clean_signed` picks the most fragile set of neurons.** Its top-*positive* clean-side neurons are the ones that most increase the KL away from the noise prediction. Steering them in *either* direction hurts: on IEMOCAP, accuracy goes from 0.693 to 0.23–0.29 at alpha < 0 and to 0.63 even at alpha = +0.3. This makes `clean_signed` useful for **ablation and diagnosis**, but not for improving answers.
* **Absolute-value scores (`clean_abs`, `noise_abs`) and `aad` are much less destructive** and give the small gains. `noise_abs` has the mildest alpha = −1 damage on IEMOCAP (for example, `per_token` 0.667 versus 0.226 for `clean_signed`).
* **For `final` mode, the gradient-free `aad` baseline is the best score on both datasets** (MMAR +3.0, IEMOCAP +1.7 at alpha = −1). Under KL, the gradient does not improve the final-position intervention over the plain activation difference.
* **`per_token` selection does not beat shared selection.** With the same per-position budget, it is about equal to or worse than `audio` mode (MMAR `aad`: +0.3 versus +0.3; IEMOCAP `clean_abs`: +0.1 versus +0.3) and is more destructive at alpha = −1 for most scores.
* **MMSU does not move under KL either.** No setting improves on alpha = 0, which agrees with Experiment 1.

### 4.3 Engineering checks

* **Batched alphas.** `selftest_alpha_batch` on MMAU-mini (`text`, K = 1k) returned identical strings for all 5 alphas whether generated in one batch or one by one, so `BATCH_ALPHAS = True` is safe.
* **FAST_ATTR (dAB on GPU).** Over 3 items, the selected neuron sets differed by 2 neurons at the top-K boundary, and 0 of 15 predicted letters changed. The option is kept off for bit-level consistency with earlier runs.
* **Refactor equivalence.** For this refactor, the original `partb.py` and the new package were both run on a small CPU stand-in model with the same hook structure. Every selection score, both settings of `PERTOKEN_TOPK`, all 5 modes, 2 K values, and the batched and unbatched alpha paths gave **identical** attribution scores, specs, and all 540 prediction files.
* **Gradient-norm diagnostic.** On the first MMAR item (T = 810), the per-position gradient L2 norm is 0.015 on audio positions and 0.37 on text positions. On MMSU (T = 143) it is 0.029 versus 0.18, and on IEMOCAP (T = 102) 0.022 versus 0.069. The KL objective reacts about 3–25× more strongly per text position than per audio position.

---

## 5. Discussion

1. **The attribution localizes something causal.** Ablating the selected neurons toward the noise run produces large, monotone, K-dependent accuracy drops on all six datasets. The size of the drop depends on the selection score in an interpretable way: sign-aware clean-side selection is the most damaging. This is the strongest result so far and fits an *analysis* paper (where audio information enters, at which positions, and how much of the decision depends on it) better than an *improvement* paper.
2. **Gains are small and come from one specific intervention.** The repeatable gains (2–3 points on MMAR, about 1–2 on IEMOCAP and RAVDESS) come from `final` mode with alpha < 0, which partially *undoes* the audio-induced shift at the answer position. Explaining this, for example with an output-bias hypothesis, is a promising next analysis. The gains do not appear on MMAU-mini, MMSU, or MELD.
3. **The gradient's value is not yet established.** On the two datasets where scores can be compared under KL, the gradient-free `aad` matches or beats the gradient-based scores for the intervention that helps (`final`). The gradient does matter for *which* neurons are fragile (`clean_signed` versus the others). To support the proposed `contrast` score, the next step is to run it under KL on MMAR and IEMOCAP and compare it with `aad` and a random-neuron control.

---

## 6. Limitations and threats to validity

* **Oracle alpha.** "best Δ" picks the best of 4 non-zero alphas on the test set. Under the null hypothesis, the maximum of 4 noisy differences is biased upward. The alpha (and mode and K) should be chosen on a held-out split, or one setting should be fixed in advance, for example `final`, alpha = −1, any K.
* **No significance testing yet.** With n = 1000 (MMAR), a 2–3 point difference may or may not be significant, depending on how many items flip. `caga.results.paired_test` (exact McNemar test on the same items) has been added, and `notebooks/03_analyze_results.ipynb` runs it on every configuration. It needs the per-item prediction files from the original result directories.
* **Environment variance.** Both experiments ran on an A100 80GB, but in different software environments (for example Python 3.12 versus 3.13, with different library versions). The same unsteered model (alpha = 0) scored 0.623 on MMAR in Experiment 1 and 0.629 in Experiment 2 (MMSU 0.6502 versus 0.6506, IEMOCAP 0.6934 versus 0.6927). Kernel and library differences shift greedy outputs by up to about 0.6 points, which is comparable to many of the reported effects. Comparisons should be made only within one environment, and library versions should be recorded with each run.
* **Incomplete comparisons.** The two experiments differ in loss (TV probably, versus KL) and in software environment. In Experiment 2, `contrast` was not run on MMAR or IEMOCAP. MELD (Experiment 1) is 39% complete. CREMA-D and AIR-Bench have not been run.
* **No random-neuron or random-direction control.** Without one, we cannot say how much of the alpha = −1 damage is specific to the selected neurons rather than to perturbing any K neurons by noise-run differences.
* **Noise seed depends on the file path.** The Gaussian noise for an item is seeded from its *absolute audio path*, so the same item gets different noise whenever the dataset is stored under a different directory, and attributions are not exactly reproducible across machines. `config.CONTRAST_SEED_KEY = "id"` fixes this for new runs. The default stays `"path"` so that existing runs can be resumed consistently.
* **Single contrast type.** Only white Gaussian noise at 0 dB was used. Other contrasts (silence, a different clip, shuffled frames, or noise at other SNRs) may isolate different information.
* **Partial gradients and scope sums.** See Section 2.3. Blocking downstream gate paths and summing signed salience over positions before ranking are both design choices whose effect has not been measured.
* **Text scope includes the template.** `text` positions include the system prompt and chat-template tokens, not just the question and options.
* **IEMOCAP and MELD protocols.** The IEMOCAP set is a public-mirror approximation (5528 versus the canonical 5531). MELD uses dev + test with all 7 classes, so it is not a standard weighted-F1 setup.

---

## 7. Recommended next steps

1. **Close the selection-score table under KL.** Run `contrast` and `noise_abs` on MMAR, and `contrast` and `noise_signed` on IEMOCAP:
   `python scripts/run_steering.py mmar --select-score contrast noise_abs --eval`
2. **Add a random-neuron control.** Use the same K, the same positions, and the same `dAB` injection, with uniformly random neurons. This is the baseline needed to claim that the selection matters.
3. **Report paired tests and a fixed protocol.** Decide on one setting in advance (for example, `final`, alpha = −1, K = 1k) and report McNemar p-values per dataset. Alternatively, tune alpha on half of each dataset and test on the other half.
4. **Look into the `final`-mode effect.** Check which answer options gain and lose (the confusion matrix before and after), whether the effect is concentrated in a few neurons (K = 10, 50, 100), and in which layers.
5. **Finish the coverage.** Complete MELD, then run CREMA-D and AIR-Bench Foundation (optionally with `MAX_PER_TASK`).
6. **Make runs portable.** Switch to `CONTRAST_SEED_KEY = "id"` for new experiments, and record the loss and library versions with each run.

---

## Appendix A: Version notes

* **v30.1:** per-item attribution with an IG estimator option, TV loss, CREMA-D evaluation set.
* **v30.2:** IG removed (gradient × activation only), multi-benchmark manifests, `per_token` mode, six selection scores, KL loss, alpha batching and other speed-ups.
* **Refactor (this version):** the single `partb.py` was split into `model / contrast / attribution / selection / steering / experiment / runs / results / selftest`, with numerically identical behavior (Section 4.3). Changes:
  * all settings were moved to `caga/config.py`;
  * the broken `scripts/download.py` import was fixed;
  * the evaluator now counts unique ids instead of lines;
  * added: the `CONTRAST_SEED_KEY` option, the McNemar test, CLI flags for selection score, modes, K, and per-token top-K, and torch-free result aggregation;
  * all Chinese text in code and notebooks was translated into English.

## Appendix B: Where the numbers live

| file | content |
|---|---|
| `results/exp1_k_sweep_contrast.csv` | 144 rows: 6 datasets × 4 modes × 6 K, all alphas, completeness status |
| `results/exp2_select_score_ablation_kl.csv` | 45 rows: dataset × select score × mode at K = 10k, all alphas |
| `legacy/exp1_k_sweep_contrast.ipynb` | original Experiment 1 notebook with outputs (translated) |
| `legacy/exp2_select_score_ablation_kl.ipynb` | original Experiment 2 notebook with outputs (translated) |
| `$CAGA_RUNS/caga_eval_v30_2/<dataset>/…/alpha_*.jsonl` | per-item predictions (in the original result directories, not in this folder) |
