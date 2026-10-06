#!/usr/bin/env python3
"""Run per-item attribution + activation steering on one or more datasets (GPU, >= 24 GB).

    python scripts/run_steering.py mmau_mini --eval
    python scripts/run_steering.py mmar --select-score contrast aad --modes text audio --k 10000
    python scripts/run_steering.py mmar --per-token-topk --modes audio text audio_text --eval

Runs are resumable: finished (dataset, select score, mode, K, alpha, item) entries are skipped.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caga import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"default: {config.DATASETS}")
    ap.add_argument("--select-score", nargs="+", default=[config.SELECT_SCORE], choices=config.SELECT_SCORES,
                    help="one or more neuron ranking scores, run one after another")
    ap.add_argument("--modes", nargs="+", default=config.MODES, choices=config.ALL_MODES)
    ap.add_argument("--k", nargs="+", type=int, default=config.K_LIST, help="number of steered neurons")
    ap.add_argument("--alphas", nargs="+", type=float, default=config.ALPHAS)
    ap.add_argument("--per-token-topk", action="store_true", help="each position selects its own top-K")
    ap.add_argument("--no-batch-alphas", action="store_true", help="generate each alpha separately")
    ap.add_argument("--eval", action="store_true", help="print accuracy tables after running")
    ap.add_argument("--excel", action="store_true", help="write <dataset>_results.xlsx to RUNS_BASE")
    args = ap.parse_args()

    config.MODES = args.modes
    config.K_LIST = args.k
    config.ALPHAS = args.alphas
    config.PERTOKEN_TOPK = args.per_token_topk
    config.BATCH_ALPHAS = not args.no_batch_alphas
    datasets = args.datasets or config.DATASETS

    from caga.experiment import evaluate, run_steering
    for sel in args.select_score:
        config.SELECT_SCORE = sel
        print(f"\n########## SELECT_SCORE = {sel} ##########")
        run_steering(datasets)
        if args.eval:
            evaluate(datasets)
    if args.excel:
        from caga.results import save_excel
        for ds in datasets:
            save_excel(ds)


if __name__ == "__main__":
    main()
