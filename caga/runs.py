"""Result file layout and readers (no torch, usable on any machine).

    RUNS_BASE/caga_eval_v30_2/<dataset>/<config dir>/alpha_<tag>.jsonl
    one line per item: {"id", "raw", "pred", "gold", "correct"}

<config dir> = "{ESTIMATOR}[_{SELECT_SCORE}]_{mode}_{NORM}_K{K}[_pt]"; the select score is omitted
for "contrast" and "_pt" marks PERTOKEN_TOPK. The layout matches v30.2 so earlier runs resume.
"""
import json
import os

from caga import config
from caga.mcq import alpha_tag
from caga.paths import RUNS_BASE


def dataset_results_dir(dataset):
    return os.path.join(RUNS_BASE, config.RESULTS_SUBDIR, dataset)


def config_dir_name(mode, K, select_score=None, pertoken_topk=None):
    select_score = select_score or config.SELECT_SCORE
    pertoken_topk = config.PERTOKEN_TOPK if pertoken_topk is None else pertoken_topk
    pt = "_pt" if (pertoken_topk and mode != "final") else ""
    sel = "" if select_score == "contrast" else f"_{select_score}"
    return f"{config.ESTIMATOR}{sel}_{mode}_{config.NORM}_K{K}{pt}"


def result_file(dataset, mode, K, alpha):
    return os.path.join(dataset_results_dir(dataset), config_dir_name(mode, K), f"alpha_{alpha_tag(alpha)}.jsonl")


def read_results(path):
    """{item id: record} from a predictions file; duplicate ids keep the first record and corrupt lines
    (e.g. from an interrupted write) are skipped."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.setdefault(r["id"], r)
    return out


def accuracy(path):
    """(n_items, n_correct) of a predictions file, counting each item id once."""
    recs = read_results(path)
    return len(recs), sum(bool(r.get("correct")) for r in recs.values())
