"""Experiment driver: attribution -> steering -> predictions on disk (see caga.runs for the layout).

Runs are resumable at item granularity: items already present in a predictions file are skipped.
"""
import json
import os
import time

from caga import config
from caga.attribution import compute_attribution
from caga.contrast import load_audio_pair
from caga.data.common import manifest_file
from caga.mcq import alpha_tag, build_user_text, get_items, gold_letter, parse_pred
from caga.model import build_inputs, load_model
from caga.runs import accuracy, read_results, result_file
from caga.selection import build_specs
from caga.steering import Steerer


# ---------------------------------------------------------------- run
def run_steering(datasets=None):
    """Run attribution + steering for every item, mode, K and alpha in the current config."""
    datasets = datasets or config.DATASETS
    load_model()
    for ds in datasets:
        if not os.path.exists(manifest_file(ds)):
            print(f"[skip] {ds}: no manifest")
            continue
        _run_dataset(ds)


def _run_dataset(ds):
    items = get_items(ds)
    print(f"\n=== {ds}: {len(items)} items | select={config.SELECT_SCORE} | modes={config.MODES} | "
          f"K={config.K_LIST} | alphas={config.ALPHAS} | per-token top-K={config.PERTOKEN_TOPK} ===")

    files, done = {}, {}
    for mode in config.MODES:
        for K in config.K_LIST:
            for a in config.ALPHAS:
                key = (mode, K, a)
                path = result_file(ds, mode, K, a)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                done[key] = set(read_results(path))
                files[key] = open(path, "a")

    t0, n_new = time.time(), 0
    try:
        for it in items:
            if all(it["id"] in done[k] for k in files):
                continue
            user_text = build_user_text(it)
            gold = gold_letter(it)
            n_choices = len(it["choices"])

            clean, noise, _ = load_audio_pair(it)
            attr = compute_attribution(clean, noise, user_text)
            gen_inputs = build_inputs(clean, user_text)     # built once, reused for every generation
            if n_new == 0:
                d = attr["diag"]
                print(f"  [sanity] id={it['id']} grad-L2 audio/pos={d['audio']['l2_per_pos']:.5f} "
                      f"text/pos={d['text']['l2_per_pos']:.5f} final={d['final']['l2_total']:.3f} T={attr['T']}")

            for K in config.K_LIST:
                for mode in config.MODES:
                    todo = [a for a in config.ALPHAS if it["id"] not in done[(mode, K, a)]]
                    if not todo:
                        continue
                    steerer = Steerer(build_specs(attr, mode, K), attr["T"])
                    raws = steerer.generate_all(gen_inputs, todo)
                    for a, raw in zip(todo, raws):
                        pred = parse_pred(raw, n_choices)
                        rec = {"id": it["id"], "raw": raw, "pred": pred, "gold": gold, "correct": pred == gold}
                        files[(mode, K, a)].write(json.dumps(rec, ensure_ascii=False) + "\n")
                        files[(mode, K, a)].flush()
                        done[(mode, K, a)].add(it["id"])
            del attr
            n_new += 1
            if n_new % 10 == 0:
                print(f"    [{ds}] {n_new} items ({time.time() - t0:.0f}s)")
    finally:
        for f in files.values():
            f.close()
    print(f"[{ds}] done this session: {n_new} new items")


# ---------------------------------------------------------------- evaluate
def evaluate(datasets=None):
    """Print accuracy per (mode, K, alpha) for the current SELECT_SCORE / PERTOKEN_TOPK.

    best(a) is the best non-zero alpha and is chosen on the evaluation set itself (oracle)."""
    datasets = datasets or config.DATASETS
    for ds in datasets:
        if not os.path.exists(manifest_file(ds)):
            continue
        full_n = len(get_items(ds))
        print(f"\n=== {ds} | select={config.SELECT_SCORE} | norm={config.NORM} | n={full_n} ===")
        header = "mode".ljust(12) + "K".rjust(7) + \
            "".join(f"a={alpha_tag(a)}".rjust(11) for a in config.ALPHAS) + "   best(a)        dvs0    status"
        print(header)
        print("-" * len(header))
        for mode in config.MODES:
            for K in config.K_LIST:
                acc, ns = {}, {}
                for a in config.ALPHAS:
                    n, nc = accuracy(result_file(ds, mode, K, a))
                    ns[a] = n
                    acc[a] = nc / n if n else float("nan")
                complete = all(ns[a] == full_n for a in config.ALPHAS)
                status = "OK" if complete else f"PARTIAL {min(ns.values())}/{full_n}"
                base = acc.get(0.0, float("nan"))
                nonzero = [a for a in config.ALPHAS if a != 0.0 and ns[a] == full_n]
                if complete and nonzero:
                    best = max(nonzero, key=lambda a: acc[a])
                    best_s, dv = f"{acc[best]:.4f}@{best:+.1f}", f"{acc[best] - base:+.4f}"
                else:
                    best_s, dv = "-", "-"
                cells = "".join((f"{acc[a]:.4f}" if ns[a] else "  --  ").rjust(11) for a in config.ALPHAS)
                print(mode.ljust(12) + str(K).rjust(7) + cells + f"   {best_s:>12}  {dv:>8}   {status}")
            print()
