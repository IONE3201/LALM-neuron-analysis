# Legacy notebooks (v30.2)

Kept for reference only. The maintained code is the `caga/` package at the repository root, which
reproduces the original v30.2 pipeline exactly (see `../REPORT.md`, Section 4.3). Both notebooks were
run on a single NVIDIA A100 80GB GPU; storage paths in the recorded outputs are shown as `$CAGA_BASE`.

| file | content |
|---|---|
| `exp1_k_sweep_contrast.ipynb` | Experiment 1: K sweep with the `contrast` score on 6 datasets, with outputs |
| `exp2_select_score_ablation_kl.ipynb` | Experiment 2: selection-score ablation with the KL loss, with outputs |

These notebooks import the original `caga.partb` module, which has been replaced; they are records
of the runs, not runnable code. The numbers they print are collected in `../results/*.csv` and
discussed in `../REPORT.md`.
