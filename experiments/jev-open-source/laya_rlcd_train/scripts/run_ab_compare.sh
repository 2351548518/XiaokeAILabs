#!/usr/bin/env bash
# A/B 对照：同一数据、同一 seed、同一一切超参，唯一变量是 loss。
#   rlcd    = notebook 配方（GRPO 策略梯度项 + soft CE）
#   soft-ce = 纯蒸馏（soft cross-entropy 单独）
# 串行执行（两臂同时跑会抢 GPU 并引入系统噪声，破坏对照公平性）。
#
# 用法（laya-rlcd-env，实验目录下）：
#   bash scripts/run_ab_compare.sh
# 产物：runs/rlcd/ 与 runs/soft-ce/ 各自的 checkpoint + train_report.json，
#       最后打印并排对比表。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
export HF_HUB_DISABLE_XET=1

JEV_PYTHON="${JEV_PYTHON:-python}"
DATA="${JEV_TRAIN_DATA:-data/typed_decisions_train.jsonl}"
EVAL="${JEV_EVAL_DATA:-data/typed_decisions_test.jsonl}"
RUN_ROOT="${JEV_RUN_ROOT:-runs}"
LOG_DIR="${JEV_LOG_DIR:-docs/执行日志}"

# Keep prerequisites portable: no private shell scripts or interpreter paths.
# Check before creating outputs so a missing dependency cannot leave a fake run.
"$JEV_PYTHON" -c 'from laya.train import TrainConfig, finetune; import torch; assert torch.cuda.is_available(), "CUDA is required"'
for INPUT in "$DATA" "$EVAL"; do
  [[ -f "$INPUT" ]] || { echo "Missing data: $INPUT" >&2; exit 1; }
done
for ARM in rlcd soft-ce; do
  [[ ! -e "$RUN_ROOT/$ARM" && ! -e "$LOG_DIR/$ARM.log" ]] || {
    echo "Existing results for $ARM; choose new JEV_RUN_ROOT and JEV_LOG_DIR" >&2; exit 1;
  }
done
[[ ! -e "$LOG_DIR/ab-summary.log" ]] || {
  echo "Existing summary; choose a new JEV_LOG_DIR" >&2; exit 1;
}
mkdir -p "$RUN_ROOT" "$LOG_DIR"

for ARM in rlcd soft-ce; do
  LOSS="$ARM"
  echo "=== arm: $ARM (loss=$LOSS) ==="
  "$JEV_PYTHON" scripts/run_finetune.py \
    --data "$DATA" --eval-data "$EVAL" \
    --output-dir "$RUN_ROOT/$ARM" \
    --loss "$LOSS" --epochs 4 --seed 0 --device cuda \
    2>&1 | tee "$LOG_DIR/$ARM.log"
done

echo
echo "=== A/B 对比（held-out 2000 题，唯一变量 loss）==="
"$JEV_PYTHON" - "$RUN_ROOT" <<'EOF' | tee "$LOG_DIR/ab-summary.log"
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def report(arm):
    # train_report.json 是扁平结构：eval_mode/before/after 在顶层，无 train_report 外层键
    with (root / arm / "train_report.json").open() as f:
        return json.load(f)

r, s = report("rlcd"), report("soft-ce")
keys = [("accuracy", "Accuracy"), ("ece", "ECE"), ("brier", "Brier"),
        ("brier_top1", "Brier top-1"), ("mean_confidence", "Mean conf")]
print(f"{'metric':<16}{'rlcd':>12}{'soft-ce':>12}{'delta':>12}")
for k, name in keys:
    a, b = r["after"].get(k), s["after"].get(k)
    d = (a - b) if (a is not None and b is not None) else None
    fmt = lambda value: "n/a" if value is None else f"{value:.4f}"
    print(f"{name:<16}{fmt(a):>12}{fmt(b):>12}{fmt(d):>12}")
# Evaluation loss is not necessarily the final epoch's training loss.
print(f"{'eval loss':<16}{fmt(r['after'].get('loss')):>12}{fmt(s['after'].get('loss')):>12}")
print("\nby type (accuracy):")
for t in ("choice", "score", "noul"):
    ra = r["after"]["by_type"][t]["accuracy"]
    sa = s["after"]["by_type"][t]["accuracy"]
    print(f"  {t:<10}{ra:>12.4f}{sa:>12.4f}{ra - sa:>+12.4f}")
EOF
