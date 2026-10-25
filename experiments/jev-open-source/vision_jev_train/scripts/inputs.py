"""Identical image/text preprocessing for all vision entry points."""
import base64
import io
import json

from PIL import Image

INPUT_SCHEMA = "scienceqa.question_only.v2"
PLACE = "<|vision_start|><|image_pad|><|vision_end|>"
MAX_LEN = 320
MAX_PIXELS = 320 * 320


def load_rows(path):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty split: {path}")
    # Old prepared files included answer explanations. Reject them rather than
    # silently reproducing an evaluation with privileged input information.
    if any(r.get("input_schema") != INPUT_SCHEMA for r in rows):
        raise ValueError(f"Regenerate {path} with the current prepare_data.py")
    return rows


def load_image(b64):
    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    width, height = image.size
    if max(width, height) > 320:
        ratio = 320 / max(width, height)
        image = image.resize(
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            Image.Resampling.BILINEAR,
        )
    return image


def build_text(state, question, option):
    return f"{PLACE}State:\n{state}\n\nQuestion:\n{question}\n\nOption:\n{option}"


def encode_batch(proc, rows, max_len=MAX_LEN):
    texts, images, counts = [], [], []
    for row in rows:
        image = load_image(row["image_png_b64"])
        # Token-level truncation can cut expanded image markers. Bound state
        # before rendering, preserving its trailing hint, and keep markers whole.
        state = (row.get("state") or "")[-max_len * 4 :]
        for option in row["options"]:
            texts.append(build_text(state, row["question"], option))
            images.append(image)
        counts.append(len(row["options"]))
    return (
        proc(text=texts, images=images, return_tensors="pt", padding=True, max_pixels=MAX_PIXELS),
        counts,
    )


def model_inputs(enc):
    keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
    return {key: enc[key] for key in keep if key in enc}
