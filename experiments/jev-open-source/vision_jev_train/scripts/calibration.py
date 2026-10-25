"""Shared temperature contract for training, evaluation and serving."""
import json
import math
from pathlib import Path
import warnings

import torch


def validate_temperature(value):
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    return temperature


def probabilities(logits, temperature=1.0):
    """Accept raw logits only; apply the temperature exactly once here."""
    return torch.softmax(logits.float() / validate_temperature(temperature), dim=-1)


def fit_temperature(logits_pool, rows):
    if not rows or len(logits_pool) != len(rows):
        raise ValueError("calibration requires nonempty, aligned logits and labels")
    golds = [torch.tensor(r["gold"], dtype=torch.float32) for r in rows]
    golds = [g / g.sum() for g in golds]
    candidates = []
    # Use the same grid for both entry points. A boundary winner is a warning,
    # not evidence that the unconstrained optimum has been found.
    for k in range(41):
        t = 0.5 + k * 0.1
        ce = sum(
            -(g * probabilities(lg, t).clamp_min(1e-9).log()).sum().item()
            for lg, g in zip(logits_pool, golds)
        ) / len(rows)
        candidates.append((ce, t))
    ce, t = min(candidates)
    if t in (0.5, 4.5):
        warnings.warn("Calibration optimum is at the grid boundary; inspect calibration data.")
    return t, ce


def save_temperature(path, temperature, calibration_n, soft_ce):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "temperature": validate_temperature(temperature),
        "calibration_n": calibration_n,
        "soft_ce": soft_ce,
        "formula": "softmax(raw_logits / temperature)",
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_temperature(checkpoint, override=None, path=None):
    if override is not None:
        return validate_temperature(override)
    # Training stores calibration beside the adapter. Evaluation can export a
    # separate file; an explicit path avoids silently using an older fit.
    source = Path(path) if path else Path(checkpoint) / "temperatures.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing calibration: {source}. Pass --temperatures FILE or "
            "--temperature 1 explicitly for uncalibrated inference."
        )
    return validate_temperature(json.loads(source.read_text())["temperature"])
