"""Turn an attribution into steering specs: which (layer, neuron) pairs to steer, at which positions,
and by how much.

A spec is {"idx": {layer: LongTensor[n]}, "delta": {layer: FloatTensor[T, n]}}; at steering time the
hook adds alpha * delta[layer] to the activations of neurons idx[layer] during prefill.
"""
import torch

from caga import config
from caga.model import STATE


def ranks_by_abs():
    """contrast / *_abs rank by |score|; *_signed and aad take the top positive scores."""
    return config.SELECT_SCORE in ("contrast", "clean_abs", "noise_abs")


def _max_k(K=None):
    return max(max(config.K_LIST), K or 0)


def rank_neurons(score_ld, max_k):
    """Flat indices (layer * D + neuron) of the top `max_k` entries of an [L, D] score, descending.
    The top-K set for any K <= max_k is the prefix order[:K]."""
    flat = (score_ld.abs() if ranks_by_abs() else score_ld).reshape(-1)
    return flat.topk(min(max_k, flat.numel())).indices


def group_by_layer(flat_idx):
    d_int = STATE.d_int
    layers = torch.div(flat_idx, d_int, rounding_mode="floor")
    neurons = flat_idx % d_int
    return {int(li): neurons[layers == li].to(torch.long) for li in torch.unique(layers).tolist()}


def _normalize(delta, norm):
    if norm == "none":
        return delta
    ssq = sum(float((d.double() ** 2).sum()) for d in delta.values())
    g = (ssq ** 0.5) or 1.0
    return {li: d / g for li, d in delta.items()}


def spec_shared(order, dab, positions, T, K, norm):
    """One neuron set (top-K of `order`) shared by all `positions`; each position injects its own dAB."""
    per_layer = group_by_layer(order[:K])
    dev = dab[0].device if len(dab) else order.device
    pos = torch.tensor(sorted(positions), dtype=torch.long, device=dev)
    delta = {}
    for li, idx in per_layer.items():
        idx = idx.to(dev)
        per_layer[li] = idx
        d = torch.zeros(T, idx.numel(), dtype=torch.float32, device=dev)
        if pos.numel():
            d[pos] = dab[li][pos][:, idx]
        delta[li] = d
    return {"idx": per_layer, "delta": _normalize(delta, norm)}


def _per_token_order(attr, positions, key, K):
    """Top-maxK flat indices for every position (batched), cached per item and position set."""
    cache = attr.setdefault("_pt_order", {})
    if key not in cache:
        dev = attr["dab"][0].device
        pos = torch.tensor(sorted(positions), dtype=torch.long)
        score = attr["score_pt"][:, pos, :]
        score = (score.abs() if ranks_by_abs() else score).permute(1, 0, 2).reshape(pos.numel(), -1).to(dev)
        top = score.topk(min(_max_k(K), score.shape[1]), dim=1).indices      # [P, maxK]
        del score
        cache[key] = (pos.to(dev), top)
    return cache[key]


def spec_per_token(attr, positions, T, K, key, norm):
    """Every position picks its own top-K (layer, neuron) pairs from its own salience and injects its
    dAB on them. Same per-position budget as spec_shared, but different neurons per position."""
    dab = attr["dab"]
    dev = dab[0].device
    if len(positions) == 0:
        return {"idx": {}, "delta": {}}
    pos, top = _per_token_order(attr, positions, key, K)
    top = top[:, :K]                                                  # [P, K]
    P, kk = top.shape
    layer_idx = torch.div(top, STATE.d_int, rounding_mode="floor")
    neuron_idx = top % STATE.d_int
    pos_grid = pos.view(P, 1).expand(P, kk)
    idx_out, delta = {}, {}
    for li in torch.unique(layer_idx).tolist():
        hit = layer_idx == li
        n_sel, p_sel = neuron_idx[hit], pos_grid[hit]
        uniq, inv = torch.unique(n_sel, return_inverse=True)        # union of neurons in this layer
        d = torch.zeros(T, uniq.numel(), dtype=torch.float32, device=dev)
        d[p_sel, inv] = dab[li][p_sel, n_sel].float()
        idx_out[int(li)] = uniq.to(torch.long)
        delta[int(li)] = d
    return {"idx": idx_out, "delta": _normalize(delta, norm)}


def build_specs(attr, mode, K, norm=None):
    """Steering specs for one (mode, K). See REPORT.md, section 2.5, for the modes."""
    norm = norm or config.NORM
    dab, T = attr["dab"], attr["T"]
    aidx, tidx = attr["audio_idx"], attr["text_idx"]
    positions = {"final": [T - 1], "all": list(range(T)), "audio": aidx, "text": tidx}

    if mode == "per_token":
        if attr.get("score_pt") is None:
            raise RuntimeError("per_token mode needs score_pt; include 'per_token' in config.MODES "
                               "before computing the attribution.")
        return [spec_per_token(attr, aidx, T, K, "audio", norm)]

    if config.PERTOKEN_TOPK and mode != "final":
        if attr.get("score_pt") is None:
            raise RuntimeError("PERTOKEN_TOPK is on but the attribution has no score_pt; recompute it.")
        if mode == "audio_text":
            return [spec_per_token(attr, aidx, T, K, "audio", norm),
                    spec_per_token(attr, tidx, T, K, "text", norm)]
        return [spec_per_token(attr, positions[mode], T, K, mode, norm)]

    cache = attr.setdefault("_order", {})        # rank each scope once per item, reuse for every K

    def order(scope):
        if scope not in cache:
            cache[scope] = rank_neurons(attr["score"][scope], _max_k(K))
        return cache[scope]

    if mode == "audio_text":
        return [spec_shared(order("audio"), dab, aidx, T, K, norm),
                spec_shared(order("text"), dab, tidx, T, K, norm)]
    return [spec_shared(order(mode), dab, positions[mode], T, K, norm)]
