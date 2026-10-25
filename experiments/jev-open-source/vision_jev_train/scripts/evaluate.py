#!/usr/bin/env python
"""独立评测脚本：对已训练好的 LoRA adapter 做完整评测。

与训练、服务共用图像预缩放和文本构造。每个划分只计算一次原始 logits，
温度只在归一化时应用一次，校准结果另存 temperatures.json 供服务显式加载。

用法：
  python evaluate.py --model models/Qwen3.5-2B --ckpt runs/v1/final \
      --data prepared --out runs/v1/eval
"""
import argparse, json, os, time
from pathlib import Path
import torch
from calibration import probabilities, fit_temperature, save_temperature, validate_temperature
from inputs import load_rows, encode_batch, model_inputs


def log(*a):
    print(*a, flush=True)


def soft_ce(probs, gold):
    return -(gold * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()


def brier(probs, gold):
    return ((probs - gold) ** 2).sum(dim=-1).mean()


@torch.no_grad()
def run_logits(proc, model, rows, device, bs=2):
    model.eval()
    pool = []
    for i in range(0, len(rows), bs):
        enc, counts = encode_batch(proc, rows[i : i + bs])
        enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs(enc).items()}
        out = model(**enc)
        logits = out.logits.float().squeeze(-1)
        pos = 0
        for c in counts:
            pool.append(logits[pos : pos + c].cpu())
            pos += c
    return pool


def metrics(probs_list, rows):
    accs, ces, brs = [], [], []
    confs, correct = [], []
    for p, r in zip(probs_list, rows):
        g = torch.tensor(r["gold"], dtype=torch.float32)
        gold = g / g.sum()
        hit = 1.0 if p.argmax().item() == gold.argmax().item() else 0.0
        accs.append(hit)
        ces.append(soft_ce(p, gold).item())
        brs.append(brier(p, gold).item())
        confs.append(p.max().item())
        correct.append(hit)
    n = max(len(rows), 1)
    bins = [[] for _ in range(15)]
    for c, a in zip(confs, correct):
        bins[min(int(c * 15), 14)].append((c, a))
    ece = sum(abs(sum(cc for cc, _ in b) - sum(aa for _, aa in b)) for b in bins if b) / n
    return {
        "n": len(rows),
        "acc": sum(accs) / n,
        "soft_ce": sum(ces) / n,
        "brier": sum(brs) / n,
        "ece": ece,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoProcessor, Qwen3_5ForSequenceClassification
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.ckpt)
    base = Qwen3_5ForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.bfloat16
    )
    tc = base.config.get_text_config()
    tok = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tc.pad_token_id = tok.pad_token_id
    tc.eos_token_id = tok.eos_token_id
    base.num_labels = 1
    net = PeftModel.from_pretrained(base, args.ckpt)
    net.to(device).eval()
    log(f"[load] adapter={args.ckpt} T={args.temperature}")

    splits = {}
    for name in ("dev", "calibration", "test"):
        p = os.path.join(args.data, f"{name}.jsonl")
        if os.path.exists(p):
            splits[name] = load_rows(p)
    log(f"[data] " + " ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # Cache raw logits once per split; temperature is never applied in the
    # model runner, preventing accidental softmax(logits / T**2).
    pools = {}
    best_t, best_ce = validate_temperature(args.temperature), None
    if "calibration" in splits:
        pools["calibration"] = run_logits(proc, net, splits["calibration"], device)
        best_t, best_ce = fit_temperature(pools["calibration"], splits["calibration"])
        save_temperature(
            Path(args.out) / "temperatures.json", best_t, len(splits["calibration"]), best_ce
        )
        log(f"[calib] best_T={best_t:.3f} calib_soft_ce={best_ce:.5f}")

    report = {"ckpt": args.ckpt, "temperature": best_t, "calib_soft_ce": best_ce, "splits": {}}
    t0 = time.time()
    for name, rows in splits.items():
        pool = pools[name] if name in pools else run_logits(proc, net, rows, device)
        raw = metrics([probabilities(lg) for lg in pool], rows)
        cal = metrics([probabilities(lg, best_t) for lg in pool], rows)
        report["splits"][name] = {"raw": raw, "calibrated": cal}
        log(
            f"[{name:11s}] n={raw['n']:4d} acc={raw['acc']:.4f} soft_ce={raw['soft_ce']:.4f} "
            f"brier={raw['brier']:.4f} ece={raw['ece']:.4f} | "
            f"acc_cal={cal['acc']:.4f} soft_ce_cal={cal['soft_ce']:.4f} ece_cal={cal['ece']:.4f}"
        )

    report["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(args.out, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("[done] " + json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
