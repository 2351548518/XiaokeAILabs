#!/usr/bin/env python3
"""metask-jev-4b 推理实测。单卡 CUDA。

四个关注点，都与 laya 同口径以便对比：
  A. 加载与输出契约，A-Z 单 token 契约验证
  B. 延迟基准（单次 score 的 p50/p90/p99）
  C. 中英平行对比（README 宣称 16 语言含中文，这是最该验的一条）
  D. 温度系数前后的校准差异（它宣称 ECE 0.114 -> 0.040）
"""
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
print("host:", os.uname().nodename)

import sys

sys.path.insert(0, os.path.expanduser("~/metask-jev"))

import torch
from jev_scorer import load_model, score, apply_temperature

# Resolve relative to this script, not the caller's working directory. Missing
# calibration is an error; silently falling back to T=1 changes the benchmark.
TEMP_PATH = Path(__file__).resolve().with_name("metask-temperature.json")
TEMP = json.loads(TEMP_PATH.read_text(encoding="utf-8"))
BY_KIND = TEMP["by_kind"]
print("官方温度系数:", BY_KIND, "| ECE %s -> %s" % (TEMP.get("ece_before"), TEMP.get("ece_after")))


def avail_gb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1024 / 1024


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def report(tag, xs):
    xs = sorted(xs)
    if not xs:
        print(tag, "无样本")
        return
    print(
        "%-42s n=%-4d min %8.2f  p50 %8.2f  p90 %8.2f  p99 %8.2f  max %9.2f ms"
        % (tag, len(xs), xs[0], pct(xs, 50), pct(xs, 90), pct(xs, 99), xs[-1])
    )


print("MemAvailable 基线: %.1f GB" % avail_gb())

# Versioned shared inputs keep the two classification arms aligned.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_cases import CASES as EN_CASES, CRITERIA, INSTRUCTIONS, BENCHMARK_VERSION

print("benchmark:", BENCHMARK_VERSION)


def make_schema():
    return {
        "decision": {
            "description": INSTRUCTIONS,
            "type": "enum",
            "choices": ["billing", "technical", "sales", "other"],
            "choice_descriptions": CRITERIA,
        }
    }


MODEL = "wayfind/metask-jev-4b-policy-mix"

# ============ A. 契约 ============
print("\n" + "=" * 96)
print("[A] 加载与输出契约")
print("=" * 96)
t0 = time.time()
model, tok, dev = load_model(MODEL)
print("load_model 完成 %.2f s，device=%s，当前可用 %.1f GB" % (time.time() - t0, dev, avail_gb()))
if torch.cuda.is_available():
    print(
        "CUDA 已分配: %.2f GB  峰值: %.2f GB"
        % (torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3)
    )

bad = [
    chr(65 + i) for i in range(26) if len(tok.encode(chr(65 + i), add_special_tokens=False)) != 1
]
print("A-Z 单 token 契约:", "通过" if not bad else ("损坏: %s" % bad))

r = score(
    model,
    tok,
    json.dumps({"body": EN_CASES[0][1]}, ensure_ascii=False),
    make_schema(),
    temperature=BY_KIND.get("choice", 1.0),
)
print("样例输出:", json.dumps(r, ensure_ascii=False))

# ============ B. 延迟 ============
print("\n" + "=" * 96)
print("[B] 延迟基准，单次 score，1 个 enum 字段 4 选项")
print("=" * 96)
for c in (EN_CASES[0][1], EN_CASES[5][1]):
    score(
        model,
        tok,
        json.dumps({"body": c}, ensure_ascii=False),
        make_schema(),
        temperature=BY_KIND.get("choice", 1.0),
    )
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

lat = []
for i in range(120):
    c = EN_CASES[i % len(EN_CASES)][1]
    t = time.time()
    score(
        model,
        tok,
        json.dumps({"body": c}, ensure_ascii=False),
        make_schema(),
        temperature=BY_KIND.get("choice", 1.0),
    )
    lat.append((time.time() - t) * 1000)
report("英文（官方 T=%.2f）" % BY_KIND.get("choice", 1.0), lat)

lat_raw = []
for i in range(40):
    c = EN_CASES[i % len(EN_CASES)][1]
    t = time.time()
    score(model, tok, json.dumps({"body": c}, ensure_ascii=False), make_schema(), temperature=1.0)
    lat_raw.append((time.time() - t) * 1000)
report("英文（T=1.0，未校准）", lat_raw)

# ============ C. 中英平行对比 ============
print("\n" + "=" * 96)
print("[C] 中英平行对比，与 laya 同一套测试集")
print("=" * 96)
T_CHOICE = BY_KIND.get("choice", 1.0)
for lang, idx in (("英文", 1), ("中文", 2)):
    acc_l, conf_l, wrong = 0, [], []
    lat_l = []
    for gold, *texts in EN_CASES:
        body = texts[idx - 1]
        t = time.time()
        r = score(
            model,
            tok,
            json.dumps({"body": body}, ensure_ascii=False),
            make_schema(),
            temperature=T_CHOICE,
        )
        lat_l.append((time.time() - t) * 1000)
        pred = r["prediction"]
        conf = max(r["probabilities"].values())
        conf_l.append(conf)
        if pred == gold:
            acc_l += 1
        else:
            wrong.append((gold, pred, round(conf, 3), r["probabilities"]))
    print("\n--- %s ---" % lang)
    print("  准确率    : %d/%d = %.1f%%" % (acc_l, len(EN_CASES), 100.0 * acc_l / len(EN_CASES)))
    print("  置信度均值: %.3f" % statistics.mean(conf_l))
    print(
        "  延迟      : p50 %.1f ms, min %.1f, max %.1f"
        % (statistics.median(lat_l), min(lat_l), max(lat_l))
    )
    for w in wrong:
        print("    错例 期望 %-10s 实得 %-10s 置信 %.3f" % (w[0], w[1], w[2]))
        print("           概率 %s" % {k: round(v, 3) for k, v in w[3].items()})

# ============ D. 温度前后校准对比 ============
print("\n" + "=" * 96)
print("[D] 温度系数前后，同一个分布的校准差异")
print("=" * 96)
print(
    "%-12s %-9s %-9s | %-9s %-9s"
    % ("样本", "T=1 置信", "T=%.2f 置信" % T_CHOICE, "T=1 熵", "T=%.2f 熵" % T_CHOICE)
)
for i in (0, 3, 15, 18):
    body = EN_CASES[i][1]
    r = score(
        model, tok, json.dumps({"body": body}, ensure_ascii=False), make_schema(), temperature=1.0
    )
    p1 = r["probabilities"]
    n = len(p1)
    import math

    ent1 = -sum(p * math.log(max(p, 1e-12)) for p in p1.values()) / math.log(n)
    recal = apply_temperature(p1, r["logits"], T_CHOICE)
    ent2 = -sum(p * math.log(max(p, 1e-12)) for p in recal.values()) / math.log(n)
    print(
        "%-12s %-9.3f %-14.3f | %-9.3f %-9.3f"
        % (EN_CASES[i][0], max(p1.values()), max(recal.values()), ent1, ent2)
    )

# ============ F. 资源 ============
print("\n" + "=" * 96)
print("[F] 资源占用")
print("=" * 96)
if torch.cuda.is_available():
    print("CUDA 当前分配: %.2f GB" % (torch.cuda.memory_allocated() / 1024**3))
    print("CUDA 峰值    : %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
print("MemAvailable  : %.1f GB" % avail_gb())
