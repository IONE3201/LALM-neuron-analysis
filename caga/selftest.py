"""GPU self-tests for the speed options (run once per environment before a large run)."""
import torch

from caga import config
from caga.attribution import compute_attribution
from caga.contrast import load_audio_pair
from caga.mcq import build_user_text, get_items, parse_pred
from caga.model import build_inputs, load_model
from caga.selection import build_specs
from caga.steering import Steerer


def selftest_alpha_batch(ds="mmau_mini", mode="text", K=1000):
    """Check that batched-alpha generation equals generating each alpha separately."""
    load_model()
    it = get_items(ds)[0]
    user_text = build_user_text(it)
    clean, noise, _ = load_audio_pair(it)
    attr = compute_attribution(clean, noise, user_text)
    inputs = build_inputs(clean, user_text)
    steerer = Steerer(build_specs(attr, mode, K), attr["T"])
    seq = [steerer.generate(inputs, a) for a in config.ALPHAS]
    bat = steerer.generate_batch(inputs, config.ALPHAS)
    ok = all(s == b for s, b in zip(seq, bat))
    print("RESULT:", "MATCH (batched is safe)" if ok else "MISMATCH (set config.BATCH_ALPHAS = False)")
    for a, s, b in zip(config.ALPHAS, seq, bat):
        print(f"  a={a:+.1f}  seq={s!r:20} bat={b!r:20} {'ok' if s == b else 'DIFF'}")
    return ok


def selftest_fast_attr(ds="mmau_mini", mode="text", K=1000, n_items=3):
    """Compare FAST_ATTR (dAB kept on GPU) with the CPU path: selected neurons, delta values and,
    most importantly, the predicted letters."""
    load_model()
    saved = config.FAST_ATTR
    neuron_diff, max_abs, pred_diffs, pred_total = 0, 0.0, 0, 0
    items = get_items(ds)[:n_items]
    try:
        for it in items:
            user_text = build_user_text(it)
            n_choices = len(it["choices"])
            clean, noise, _ = load_audio_pair(it)
            inputs = build_inputs(clean, user_text)
            runs = []
            for fast in (False, True):
                config.FAST_ATTR = fast
                attr = compute_attribution(clean, noise, user_text)
                specs = build_specs(attr, mode, K)
                preds = Steerer(specs, attr["T"]).generate_batch(inputs, config.ALPHAS)
                runs.append((specs, preds))
            (s0, p0), (s1, p1) = runs
            for e0, e1 in zip(s0, s1):
                for li in set(e0["idx"]) | set(e1["idx"]):
                    empty = torch.empty(0, dtype=torch.long)
                    i0 = set(e0["idx"].get(li, empty).cpu().tolist())
                    i1 = set(e1["idx"].get(li, empty).cpu().tolist())
                    neuron_diff += len(i0 ^ i1)
                    d0, d1 = e0["delta"].get(li), e1["delta"].get(li)
                    if d0 is not None and d1 is not None and d0.shape == d1.shape:
                        max_abs = max(max_abs, float((d0.cpu() - d1.cpu()).abs().max()))
            for x0, x1 in zip(p0, p1):
                pred_total += 1
                pred_diffs += parse_pred(x0, n_choices) != parse_pred(x1, n_choices)
    finally:
        config.FAST_ATTR = saved
    print(f"=== FAST_ATTR GPU vs CPU over {len(items)} items (mode={mode}, K={K}) ===")
    print(f"  selected-neuron differences : {neuron_diff}")
    print(f"  delta max absolute diff     : {max_abs:.3e}")
    print(f"  prediction differences      : {pred_diffs}/{pred_total}")
    return dict(neuron_diff=neuron_diff, max_abs=max_abs, pred_diffs=pred_diffs, pred_total=pred_total)


def time_one_item(ds="mmar", mode="text", K=1000, repeats=3):
    """Rough timing split between attribution and generation for one item."""
    import time
    load_model()
    it = get_items(ds)[0]
    user_text = build_user_text(it)
    clean, noise, _ = load_audio_pair(it)
    t = time.time()
    for _ in range(repeats):
        attr = compute_attribution(clean, noise, user_text)
    t_attr = (time.time() - t) / repeats
    inputs = build_inputs(clean, user_text)
    steerer = Steerer(build_specs(attr, mode, K), attr["T"])
    t = time.time()
    for _ in range(repeats):
        steerer.generate_batch(inputs, config.ALPHAS)
    t_gen = (time.time() - t) / repeats * len(config.K_LIST) * len(config.MODES)
    print(f"attribution: {t_attr:.2f}s | generation (all configs): {t_gen:.2f}s | "
          f"attribution share: {t_attr / (t_attr + t_gen) * 100:.0f}%")
