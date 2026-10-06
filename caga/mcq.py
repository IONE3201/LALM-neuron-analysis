"""Multiple-choice formatting: manifest items -> prompt text, gold letter, parsed prediction."""
import json
import re

from caga.data.common import manifest_file


def get_items(dataset):
    """Load all items of a dataset manifest."""
    items = []
    with open(manifest_file(dataset)) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def build_user_text(item):
    """Question, lettered options "(a) ...", and an instruction to answer with the letter only."""
    lines = [item["question"]]
    for i, c in enumerate(item["choices"]):
        lines.append(f"({chr(97 + i)}) {c}")
    lines.append("Answer with only the letter.")
    return "\n".join(lines)


def gold_letter(item):
    return chr(97 + int(item["gold_idx"]))


def parse_pred(text, n_choices):
    """Extract the predicted option letter from the model output, or None."""
    if not text:
        return None
    for pat in (r"\(\s*([a-z])\s*\)", r"^\s*([a-z])\b", r"\b([a-z])\b"):
        m = re.search(pat, text, re.I)
        if m and (ord(m.group(1).lower()) - 97) < n_choices:
            return m.group(1).lower()
    return None


def alpha_tag(a):
    """File-name-safe alpha: -1.0 -> "m1", -0.3 -> "m0p3", 0.0 -> "0", 0.3 -> "0p3"."""
    sign = "m" if a < 0 else ""
    mag = abs(a)
    s = f"{int(mag)}" if float(mag).is_integer() else f"{mag}".replace(".", "p").rstrip("0").rstrip("p")
    return f"{sign}{s or '0'}"
