"""Collect every finished configuration of a dataset into tables (pandas) and Excel files.

Replaces the aggregation cells that were copied between notebooks in v30.2.
"""
import glob
import os
import re

from caga import config
from caga.mcq import alpha_tag
from caga.paths import RUNS_BASE
from caga.runs import accuracy, config_dir_name, dataset_results_dir, read_results

_DIR_RE = re.compile(r"^(?P<est>[a-z]+)(?:_(?P<rest>.+))?_(?P<norm>none|l2)_K(?P<K>\d+)(?P<pt>_pt)?$")


def parse_config_dir(name):
    """'gradxact_aad_text_none_K10000' -> dict(select_score='aad', mode='text', K=10000, ...)."""
    m = _DIR_RE.match(name)
    if not m:
        return None
    rest = m.group("rest") or ""
    for mode in sorted(config.ALL_MODES, key=len, reverse=True):
        if rest == mode:
            sel = "contrast"
            break
        if rest.endswith("_" + mode):
            sel = rest[: -(len(mode) + 1)]
            break
    else:
        return None
    return dict(estimator=m.group("est"), select_score=sel, mode=mode, norm=m.group("norm"),
                K=int(m.group("K")), pertoken_topk=bool(m.group("pt")))


def collect(dataset, alphas=None):
    """DataFrame with one row per finished configuration directory of `dataset`."""
    import pandas as pd

    alphas = alphas or config.ALPHAS
    rows = []
    for d in sorted(glob.glob(os.path.join(dataset_results_dir(dataset), "*"))):
        info = parse_config_dir(os.path.basename(d))
        if not info or not os.path.isdir(d):
            continue
        rec = {"dataset": dataset, **info}
        acc, n0 = {}, 0
        for a in alphas:
            n, nc = accuracy(os.path.join(d, f"alpha_{alpha_tag(a)}.jsonl"))
            acc[a] = nc / n if n else float("nan")
            rec[f"a={a:+.1f}"] = round(acc[a], 4) if n else None
            if a == 0.0:
                n0 = n
        base = acc.get(0.0, float("nan"))
        nonzero = {a: v for a, v in acc.items() if a != 0.0 and v == v}
        if nonzero:
            best = max(nonzero, key=nonzero.get)
            rec["best_acc"] = round(nonzero[best], 4)
            rec["best_alpha"] = f"{best:+.1f}"
            rec["delta_vs_0"] = round(nonzero[best] - base, 4) if base == base else None
        rec["n"] = n0
        rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        sel_order = {s: i for i, s in enumerate(config.SELECT_SCORES)}
        mode_order = {m: i for i, m in enumerate(config.ALL_MODES)}
        df["_m"] = df["mode"].map(lambda m: mode_order.get(m, 99))
        df["_s"] = df["select_score"].map(lambda s: sel_order.get(s, 99))
        df = df.sort_values(["pertoken_topk", "K", "_m", "_s"]).drop(columns=["_m", "_s"]).reset_index(drop=True)
    return df


def save_excel(dataset, path=None, alphas=None):
    """Excel file with one sheet per mode (rows = select scores) plus an ALL sheet."""
    import pandas as pd

    df = collect(dataset, alphas)
    path = path or os.path.join(RUNS_BASE, f"{dataset}_results.xlsx")
    with pd.ExcelWriter(path) as xw:
        for mode in config.ALL_MODES:
            sub = df[df["mode"] == mode] if len(df) else df
            if len(sub):
                sub.to_excel(xw, sheet_name=mode[:31], index=False)
        df.to_excel(xw, sheet_name="ALL", index=False)
    print("saved:", path)
    return df


def paired_test(dataset, mode, K, alpha, select_score=None, pertoken_topk=None):
    """Exact McNemar test of steered (alpha) vs unsteered (alpha = 0) predictions on the same items.

    Returns dict(n, acc_base, acc_steered, gained, lost, p_value), where gained / lost count items that
    become correct / incorrect under steering. Only items present in both files are compared.
    """
    from math import comb

    d = os.path.join(dataset_results_dir(dataset), config_dir_name(mode, K, select_score, pertoken_topk))
    base = read_results(os.path.join(d, f"alpha_{alpha_tag(0.0)}.jsonl"))
    steer = read_results(os.path.join(d, f"alpha_{alpha_tag(alpha)}.jsonl"))
    ids = sorted(set(base) & set(steer))
    gained = sum(1 for i in ids if steer[i]["correct"] and not base[i]["correct"])
    lost = sum(1 for i in ids if base[i]["correct"] and not steer[i]["correct"])
    n_disc = gained + lost
    k = min(gained, lost)
    p = min(1.0, 2 * sum(comb(n_disc, j) for j in range(k + 1)) / 2 ** n_disc) if n_disc else 1.0
    n = len(ids)
    return dict(n=n,
                acc_base=sum(bool(base[i]["correct"]) for i in ids) / n if n else float("nan"),
                acc_steered=sum(bool(steer[i]["correct"]) for i in ids) / n if n else float("nan"),
                gained=gained, lost=lost, p_value=p)
