"""Qwen2.5-Omni Thinker loading, input construction and plain (unsteered) generation.

The loaded model is kept in the module-level `STATE` so that it is loaded only once per process.
"""
import gc
import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from caga import config


@dataclass
class ModelState:
    processor: object = None
    model: object = None
    n_layers: int = None      # number of decoder layers
    d_int: int = None         # MLP intermediate size (neurons per layer)


STATE = ModelState()


def torch_dtype():
    return getattr(torch, config.DTYPE)


def set_threads(n=None):
    """Cap CPU threads. On shared containers os.cpu_count() sees the whole host, and the default
    thread count oversubscribes the few vCPUs the container actually has, which starves the GPU."""
    n = int(n or config.CPU_THREADS)
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception as e:
        print("[threads] warn:", e)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"[threads] intra-op={torch.get_num_threads()} | affinity cores={cores}")


def seed_everything(seed=None):
    seed = config.CONTRAST_SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model():
    """Load the Qwen2.5-Omni Thinker once and return STATE."""
    if STATE.model is not None:
        return STATE
    set_threads()
    seed_everything()
    import transformers
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    transformers.logging.set_verbosity_error()

    print(f"[load] {config.MODEL_ID}")
    processor = Qwen2_5OmniProcessor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        config.MODEL_ID, torch_dtype=torch_dtype(), trust_remote_code=True)
    model.eval().to(config.DEVICE)

    tok = processor.tokenizer
    stop_ids = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None})
    model.generation_config.eos_token_id = stop_ids
    model.generation_config.pad_token_id = tok.pad_token_id or stop_ids[0]

    STATE.processor = processor
    STATE.model = model
    STATE.n_layers = len(model.model.layers)
    STATE.d_int = model.model.layers[0].mlp.up_proj.weight.shape[0]
    print(f"[load] done | layers={STATE.n_layers} | neurons/layer={STATE.d_int}")
    return STATE


def mlp_act(layer_idx):
    """The MLP activation module of a decoder layer. Its output act_fn(gate_proj(x)) is the
    per-neuron activation that is attributed and steered."""
    return STATE.model.model.layers[layer_idx].mlp.act_fn


def maybe_free():
    if config.FREE_CACHE:
        gc.collect()
        torch.cuda.empty_cache()


def build_inputs(audio, user_text, batched=True):
    """Tokenized chat input (system prompt + audio + text) on the model device, batch size 1.

    v30.2 passed the conversation unwrapped for attribution and wrapped in a list for generation;
    `batched` keeps both call forms so that results stay bit-identical to earlier runs.
    """
    conv = [
        {"role": "system", "content": [{"type": "text", "text": config.SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio},
                                     {"type": "text", "text": user_text}]},
    ]
    inputs = STATE.processor.apply_chat_template([conv] if batched else conv, add_generation_prompt=True,
                                                 tokenize=True, return_dict=True, return_tensors="pt",
                                                 padding=True)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(STATE.model.device)
    return inputs


def repeat_inputs(inputs, batch_size):
    """Replicate batch-1 inputs to `batch_size` rows (used to batch several alphas)."""
    if batch_size == 1:
        return dict(inputs)
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == 1:
            out[k] = v.repeat(batch_size, *([1] * (v.ndim - 1)))
        else:
            out[k] = v
    return out


@torch.inference_mode()
def generate(inputs, max_new=None):
    """Greedy decoding; returns one decoded string per batch row."""
    max_new = max_new or config.MAX_NEW_TOKENS
    plen = inputs["input_ids"].shape[1]
    out_ids = STATE.model.generate(**dict(inputs), max_new_tokens=max_new, do_sample=False)
    return STATE.processor.batch_decode(out_ids[:, plen:], skip_special_tokens=True)
