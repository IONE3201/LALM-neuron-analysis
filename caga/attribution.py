"""Per-item contrastive attribution over MLP neurons.

For one item we run the clean input A and the noise input B (same prompt), and compute

    salA = actA * d KL(p_A || p_B) / d actA        (gradient x activation on the clean side)
    salB = actB * d KL(p_B || p_A) / d actB        (gradient x activation on the noise side)
    dAB  = actA - actB                             (per-token activation difference)

where p_X is the next-token distribution at the last prompt position and the other side's
distribution is a detached target. The neuron ranking score is chosen by config.SELECT_SCORE.
"""
import torch
import torch.nn.functional as F

from caga import config
from caga.model import STATE, build_inputs, maybe_free, mlp_act

SCOPES = ["final", "all", "audio", "text"]


# ---------------------------------------------------------------- options
def uses_gradient():
    return config.SELECT_SCORE != "aad"


def needs_per_token():
    """Per-token salience [L, T, D] is needed for PERTOKEN_TOPK or the per_token mode."""
    return config.PERTOKEN_TOPK or ("per_token" in config.MODES)


def attribution_device():
    return STATE.model.device if config.FAST_ATTR else torch.device("cpu")


# ---------------------------------------------------------------- forward passes
def kl_last_token(logits_pred, logits_target_detached):
    """KL(pred || target) between next-token distributions at the last position."""
    logp = F.log_softmax(logits_pred[0, -1, :].float(), dim=-1)
    logq = F.log_softmax(logits_target_detached[0, -1, :].float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum()


def forward_capture(inputs):
    """No-grad forward pass that records the MLP activation of every layer."""
    captured, handles = {}, []
    for li in range(STATE.n_layers):
        def hook(module, inp, out, li=li):
            captured[li] = out
            return out
        handles.append(mlp_act(li).register_forward_hook(hook))
    try:
        with torch.no_grad():
            out = STATE.model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return out.logits, captured


def locate_spans(input_ids):
    """Indices of audio placeholder tokens and of all other (text / template) tokens."""
    ids = input_ids[0].tolist() if input_ids.dim() == 2 else input_ids.tolist()
    audio_idx = [i for i, x in enumerate(ids) if x == config.AUDIO_TOKEN_ID]
    audio_set = set(audio_idx)
    text_idx = [i for i in range(len(ids)) if i not in audio_set]
    return audio_idx, text_idx


def _gradxact_side(inputs, act, logits_target, steps, ai, ti, sal_out, want_diag, sal_pt_out=None):
    """Accumulate grad x act (integrated gradients with `steps` steps; steps=1 is grad x act).

    Every layer's activation is replaced by a leaf tensor (scale * act) so that its gradient can be
    read; sal_out[scope][layer] receives the salience summed over the positions of that scope.
    Returns squared gradient norms per scope (for the sanity print).
    """
    diag = {"audio": 0.0, "text": 0.0, "final": 0.0}
    for k in range(1, steps + 1):
        scale = float(k) / steps
        captured, handles = {}, []
        for li in range(STATE.n_layers):
            def hook(module, inp, out, li=li, scale=scale, captured=captured):
                leaf = (scale * act[li]).detach().requires_grad_(True)
                captured[li] = leaf
                return leaf
            handles.append(mlp_act(li).register_forward_hook(hook))
        try:
            out = STATE.model(**inputs)
        finally:
            for h in handles:
                h.remove()
        STATE.model.zero_grad(set_to_none=True)
        kl_last_token(out.logits, logits_target).backward()
        for li in range(STATE.n_layers):
            g = captured[li].grad[0]
            sal = act[li][0] * g
            sal_out["all"][li] += (sal.sum(0) / steps).float().cpu()
            sal_out["final"][li] += (sal[-1] / steps).float().cpu()
            if ai.numel():
                sal_out["audio"][li] += (sal[ai].sum(0) / steps).float().cpu()
            if ti.numel():
                sal_out["text"][li] += (sal[ti].sum(0) / steps).float().cpu()
            if sal_pt_out is not None:
                sal_pt_out[li] += (sal / steps).float().cpu()
            if want_diag and k == steps:
                if ai.numel():
                    diag["audio"] += float((g[ai] ** 2).sum())
                if ti.numel():
                    diag["text"] += float((g[ti] ** 2).sum())
                diag["final"] += float((g[-1] ** 2).sum())
        del captured, out
        maybe_free()
    return diag


# ---------------------------------------------------------------- selection scores
def _build_score(sel, sal_a, sal_b, dab, audio_idx, text_idx):
    """Per-scope ranking score {scope: [L, D]}."""
    if sel == "aad":
        ai = torch.tensor(audio_idx, dtype=torch.long)
        ti = torch.tensor(text_idx, dtype=torch.long)
        zero = torch.zeros(STATE.d_int)

        def per_layer(reduce):
            return torch.stack([reduce(dab[li]) for li in range(STATE.n_layers)], dim=0)

        return {"final": per_layer(lambda d: d[-1]),
                "all": per_layer(lambda d: d.sum(0)),
                "audio": per_layer(lambda d: d[ai].sum(0) if ai.numel() else zero),
                "text": per_layer(lambda d: d[ti].sum(0) if ti.numel() else zero)}
    if sel in ("clean_abs", "clean_signed"):
        return {sc: sal_a[sc] for sc in SCOPES}
    if sel in ("noise_abs", "noise_signed"):
        return {sc: sal_b[sc] for sc in SCOPES}
    if sel == "contrast":
        return {sc: sal_a[sc] - sal_b[sc] for sc in SCOPES}
    raise ValueError(f"unknown SELECT_SCORE {sel!r}")


def _build_score_per_token(sel, sal_a_pt, sal_b_pt, dab):
    """Per-token ranking score [L, T, D]."""
    n = STATE.n_layers
    if sel == "aad":
        return torch.stack([dab[li] for li in range(n)], dim=0)
    if sel in ("clean_abs", "clean_signed"):
        return torch.stack([sal_a_pt[li] for li in range(n)], dim=0)
    if sel in ("noise_abs", "noise_signed"):
        return torch.stack([sal_b_pt[li] for li in range(n)], dim=0)
    return torch.stack([sal_a_pt[li] - sal_b_pt[li] for li in range(n)], dim=0)


# ---------------------------------------------------------------- entry point
def compute_attribution(clean, noise, user_text, steps=None):
    """Attribution for one item. Returns a dict with
        score      {scope: [L, D]}  ranking score summed over the scope's positions
        score_pt   [L, T, D] or None
        dab        {layer: [T, D]}  actA - actB, the vector that is injected when steering
        audio_idx, text_idx, T, diag
    """
    steps = steps or config.IG_STEPS
    n_layers, d_int = STATE.n_layers, STATE.d_int
    inputs_a = build_inputs(clean, user_text, batched=False)
    inputs_b = build_inputs(noise, user_text, batched=False)
    audio_idx, text_idx = locate_spans(inputs_a["input_ids"])
    ai = torch.tensor(audio_idx, dtype=torch.long)
    ti = torch.tensor(text_idx, dtype=torch.long)

    logits_a, cap_a = forward_capture(inputs_a)
    act_a = {li: cap_a[li].detach() for li in cap_a}
    logits_a = logits_a.detach()
    logits_b, cap_b = forward_capture(inputs_b)
    act_b = {li: cap_b[li].detach() for li in cap_b}
    logits_b = logits_b.detach()
    del cap_a, cap_b
    maybe_free()

    dab = {li: (act_a[li][0] - act_b[li][0]).float() for li in range(n_layers)}
    T = dab[0].shape[0]

    sal_a = sal_b = sal_a_pt = sal_b_pt = None
    diag = {"audio": 0.0, "text": 0.0, "final": 0.0}
    if uses_gradient():
        sal_a = {sc: torch.zeros(n_layers, d_int, dtype=torch.float32) for sc in SCOPES}
        sal_b = {sc: torch.zeros(n_layers, d_int, dtype=torch.float32) for sc in SCOPES}
        if needs_per_token():
            sal_a_pt = {li: torch.zeros(T, d_int, dtype=torch.float32) for li in range(n_layers)}
            sal_b_pt = {li: torch.zeros(T, d_int, dtype=torch.float32) for li in range(n_layers)}
        diag = _gradxact_side(inputs_a, act_a, logits_b, steps, ai, ti, sal_a, True, sal_a_pt)
        _gradxact_side(inputs_b, act_b, logits_a, steps, ai, ti, sal_b, False, sal_b_pt)

    score = _build_score(config.SELECT_SCORE, sal_a, sal_b, dab, audio_idx, text_idx)
    score_pt = _build_score_per_token(config.SELECT_SCORE, sal_a_pt, sal_b_pt, dab) if needs_per_token() else None
    dev = attribution_device()
    dab = {li: dab[li].to(dev) for li in range(n_layers)}

    n_pos = {"audio": max(len(audio_idx), 1), "text": max(len(text_idx), 1), "final": 1}
    diag = {k: {"l2_total": diag[k] ** 0.5, "l2_per_pos": (diag[k] / n_pos[k]) ** 0.5, "n": n_pos[k]}
            for k in diag}

    del act_a, act_b
    maybe_free()
    STATE.model.zero_grad(set_to_none=True)
    return {"score": score, "score_pt": score_pt, "dab": dab,
            "audio_idx": audio_idx, "text_idx": text_idx, "T": T, "diag": diag, "steps": steps}
