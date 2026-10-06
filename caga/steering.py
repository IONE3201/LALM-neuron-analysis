"""Activation steering: add alpha * delta to selected MLP neurons during the prefill pass."""
from collections import defaultdict

import torch

from caga import config
from caga.model import STATE, generate, mlp_act, repeat_inputs, torch_dtype


class Steerer:
    """Holds the steering specs of one item and registers forward hooks for given alpha(s).

    Steering is applied only during prefill (sequence length > 1); decoding steps are untouched.
    The prefill length must equal the attribution length T, since delta is position-specific.
    """

    def __init__(self, specs, T):
        self.T = T
        self.per_layer = defaultdict(list)
        for spec in specs:
            for li, idx in spec["idx"].items():
                self.per_layer[int(li)].append((idx, spec["delta"][li]))
        self.handles = []

    def _make_hook(self, li, alpha):
        """`alpha` is a float (one row) or a [B] tensor (one alpha per batch row)."""
        dtype = torch_dtype()
        entries = [(idx, d.to(dtype)) for idx, d in self.per_layer[li]]
        T = self.T

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                if h.shape[1] != T:
                    raise RuntimeError(f"prefill length {h.shape[1]} != attribution length {T}")
                a = alpha if isinstance(alpha, float) else alpha.to(h.device).to(h.dtype).view(-1, 1, 1)
                for idx, d in entries:
                    h[:, :, idx.to(h.device)] += a * d.to(h.device).to(h.dtype)[None]
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    def register(self, alpha):
        self.clear()
        if isinstance(alpha, float):
            if alpha == 0.0:
                return
        elif float(alpha.abs().sum()) == 0.0:
            return
        for li in self.per_layer:
            self.handles.append(mlp_act(li).register_forward_hook(self._make_hook(li, alpha)))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate(self, inputs, alpha):
        """Steered greedy generation for a single alpha."""
        if alpha == 0.0:
            return generate(inputs)[0]
        self.register(float(alpha))
        try:
            return generate(inputs)[0]
        finally:
            self.clear()

    def generate_batch(self, inputs, alphas):
        """Generate for all `alphas` in one batched call (one row per alpha). Rows differ only in the
        injected alpha, so greedy decoding of each row equals running that alpha alone."""
        batch = repeat_inputs(inputs, len(alphas))
        self.register(torch.tensor([float(a) for a in alphas], dtype=torch_dtype(), device=STATE.model.device))
        try:
            return generate(batch)
        finally:
            self.clear()

    def generate_all(self, inputs, alphas):
        if config.BATCH_ALPHAS and len(alphas) > 1:
            return self.generate_batch(inputs, alphas)
        return [self.generate(inputs, a) for a in alphas]
