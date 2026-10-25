"""把 ScienceQA 转成视觉决策训练用的 jsonl，并按图片内容做可复现切分。

与 text_jev_train/prepare_data.py 的差异：
1. 每条样本带一张图，图像以 base64 PNG 内嵌进 parquet，不依赖额外下载 COCO
2. 切分主键从 case id 换成图片 sha256。ScienceQA 的图在 train 与 test 之间
   可能复用，若只按行 id 切分会把同一张图
   同时放进训练集和测试集，退化成背图，指标虚高
3. 先去重，再按 seed 与图片哈希排序切分，不依赖 parquet 的行序

产出：
  prepared/{train,dev,calibration,test}.jsonl
  prepared/split_manifest.json（含每个划分的图片 sha256 全表与源文件 sha256）

每行字段：
  image_png_b64   图像 base64
  state           题面上下文（选项与提示，不含 lecture、solution 或 answer）
  question        判别问题
  options         候选列表
  gold            概率分布，gold[answer]=1.0 其余 0
"""
import argparse
import base64
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inputs import INPUT_SCHEMA

SEED = "20260925"

SPLIT_ORDER = ["test", "train", "dev", "calibration"]


def image_bytes(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def build_state(row):
    """Build inference-time context only; labels never enter semantic input."""
    # lecture/solution are explanation annotations, not guaranteed inference
    # inputs. Exclude both; answer is used solely to construct the target.
    parts = []
    choices = row.get("choices") or []
    letters = "ABCDEFGH"
    option_text = "\n".join(
        f"{letters[i] if i < len(letters) else i}. {c}" for i, c in enumerate(choices)
    )
    parts.append(f"备选答案：\n{option_text}")
    hint = (row.get("hint") or "").strip()
    if hint:
        parts.append(f"提示：{hint}")
    return "\n\n".join(parts)


def build_question(row):
    """判别问题：给定图像与上下文，选出正确备选答案。"""
    return f"{row.get('question', '')}\n" "请根据图像与上下文，选出正确的备选答案。"


def convert(row):
    """一条 ScienceQA 样本转成一条决策样本。"""
    raw = row.get("image")
    if raw is None:
        return None
    if hasattr(raw, "convert"):
        img_bytes = image_bytes(raw)
    elif isinstance(raw, dict) and raw.get("bytes"):
        from PIL import Image

        # Canonical pixels avoid treating differently encoded copies as distinct.
        img_bytes = image_bytes(Image.open(io.BytesIO(raw["bytes"])))
    else:
        return None

    choices = list(row.get("choices") or [])
    if len(choices) < 2:
        return None
    answer = row.get("answer")
    if not isinstance(answer, int) or not (0 <= answer < len(choices)):
        return None

    gold = [0.0] * len(choices)
    gold[answer] = 1.0

    return {
        "input_schema": INPUT_SCHEMA,
        "image_sha256": sha256_hex(img_bytes),
        "image_png_b64": base64.b64encode(img_bytes).decode("ascii"),
        "state": build_state(row),
        "question": build_question(row),
        "options": [str(c) for c in choices],
        "gold": [v / sum(gold) for v in gold],
        "task": row.get("task"),
        "grade": row.get("grade"),
        "subject": row.get("subject"),
        "source": "ScienceQA",
        "license": "CC BY-SA 4.0",
        "label_semantics": "multiple-choice gold; not measured action success",
    }


def split_rows(rows, seed=SEED):
    """Deduplicate first, then make nonempty 60/10/10/20 experimental splits."""
    unique = {}
    skipped_bad = skipped_dup = 0
    for row in rows:
        item = convert(row)
        if item is None:
            skipped_bad += 1
            continue
        key = item["image_sha256"]
        if key in unique:
            skipped_dup += 1
            # Stable representative even when identical images have different
            # questions. Input file ordering must not decide which case survives.
            item = min((item, unique[key]), key=lambda x: json.dumps(x, sort_keys=True))
        unique[key] = item
    items = sorted(
        unique.values(), key=lambda x: sha256_hex((str(seed) + ":" + x["image_sha256"]).encode())
    )
    if len(items) < 4:
        raise ValueError("Need at least four unique usable images for four splits")
    # Fixed quotas of 4800 could not fit a 4241-row source, even before filtering.
    # Reserve nonempty held-out splits and assign the remainder to training.
    n = len(items)
    quotas = {"test": max(1, n // 5), "dev": max(1, n // 10), "calibration": max(1, n // 10)}
    quotas["train"] = n - sum(quotas.values())
    buckets = {}
    start = 0
    for name in SPLIT_ORDER:
        buckets[name] = items[start : start + quotas[name]]
        start += quotas[name]
    return buckets, {"duplicate_image": skipped_dup, "invalid": skipped_bad}


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=ROOT / "data" / "all" / "test-00000-of-00001.parquet",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "prepared")
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()
    rows = pq.read_table(args.source).to_pylist()
    buckets, skipped = split_rows(rows, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_schema": INPUT_SCHEMA,
        "seed": args.seed,
        "source": args.source.name,
        "source_sha256": sha256_hex(args.source.read_bytes()),
        "split_policy": "deduplicated image hash, 60/10/10/20; custom experimental splits",
        "test_usage": "final evaluation only; not the original ScienceQA benchmark split",
        "excluded_input_fields": ["lecture", "solution", "answer"],
        "skipped": skipped,
        "splits": {},
    }
    for name, items in buckets.items():
        (args.out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
            encoding="utf-8",
        )
        manifest["splits"][name] = {
            "cases": len(items),
            "candidates": sum(len(item["options"]) for item in items),
            "image_sha256": sorted(item["image_sha256"] for item in items),
            "tasks": dict(Counter(item["task"] for item in items)),
        }
        print(f"[split] {name}: {len(items)} cases")
    (args.out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[skip] {skipped}")


if __name__ == "__main__":
    main()
