#!/usr/bin/env python
"""多模态 AgentJev 决策质量模型 V2：视觉底座 + LoRA + score head + 软交叉熵 + Brier + 温度校准。

本脚本包含训练更新、温度校准与内存检查：
1. 启动前检查可用统一内存，低于阈值直接拒绝启动，不进入训练
2. 训练中每步采样显存，超过 --mem-cap 立即停训、保存现场、退出，避免拖死机器
3. gradient_checkpointing 强制 use_reentrant=False（Qwen3.5 是 DeltaNet 混合架构，
   reentrant 版本在非线性结构上会静默失效或反向图错误）
4. 默认 batch 为 2，state 在拼接前限制为 1280 字符
5. 模型路径与型号可通过参数切换，便于在 0.8B/2B/4B 间选型
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import torch
from calibration import probabilities, fit_temperature, save_temperature
from inputs import load_rows, encode_batch, model_inputs

BRIER_W = 0.1


def log(*a):
    print(*a, flush=True)


_proc = None
_tok = None
_MODEL = None


def get_proc():
    global _proc, _tok
    if _proc is None:
        from transformers import AutoProcessor

        _proc = AutoProcessor.from_pretrained(_MODEL)
        _tok = _proc.tokenizer
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
    return _proc, _tok


def to_device(enc, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}


def soft_ce(probs, gold):
    return -(gold * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()


def brier(probs, gold):
    return ((probs - gold) ** 2).sum(dim=-1).mean()


def mem_gb():
    """统一内存用量（DGX Spark 无独立显存，cuda 分配即统一内存）。"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30
    return 0.0, 0.0


def avail_gb():
    """读 /proc/meminfo 的 MemAvailable，判断是否还能安全启动。"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


@torch.no_grad()
def run_logits(model, rows, device, bs=2):
    """Return raw logits; calibration belongs only in probabilities()."""
    proc, tok = get_proc()
    model.eval()
    pool = []
    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        enc, counts = encode_batch(proc, chunk)
        out = model(**model_inputs(to_device(enc, device)))
        logits = out.logits.float().squeeze(-1)
        pos = 0
        for c in counts:
            pool.append(logits[pos : pos + c].cpu())
            pos += c
    return pool


def infer_probs(model, rows, device, temperature=1.0):
    return [probabilities(lg, temperature) for lg in run_logits(model, rows, device)]


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
    from transformers import Qwen3_5ForSequenceClassification
    from peft import LoraConfig, get_peft_model

    global _MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prepared")
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--model", default=os.environ.get("JEVMODEL", "models/Qwen3.5-2B"))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--resume", default="")
    ap.add_argument("--mem-cap", type=float, default=70.0, help="统一内存硬闸 GB，超过即停训")
    ap.add_argument("--mem-min-start", type=float, default=25.0, help="启动所需最低可用内存 GB")
    ap.add_argument("--mem-min-run", type=float, default=45.0, help="运行期最低可用内存 GB，低于即停训")
    args = ap.parse_args()
    if args.steps < 1 or args.bs < 1:
        ap.error("--steps and --bs must be positive")
    _MODEL = args.model
    if Path(args.out).exists() and any(Path(args.out).iterdir()):
        raise FileExistsError("Use a new --out directory to preserve previous checkpoints")
    os.makedirs(args.out, exist_ok=True)

    # 预防第一关：启动前内存体检
    av = avail_gb()
    if av is not None:
        log(f"[mem] MemAvailable={av:.1f}GB cap={args.mem_cap}GB")
        if av < args.mem_min_start:
            log(
                f"[ABORT] 可用内存 {av:.1f}GB 低于下限 {args.mem_min_start}GB，拒绝启动。"
                "先释放内存（杀训练进程 / 停推理服务）再跑。"
            )
            sys.exit(2)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[env] device={device} torch={torch.__version__} model={_MODEL}")

    train_rows = load_rows(f"{args.data}/train.jsonl")
    dev_rows = load_rows(f"{args.data}/dev.jsonl")
    log(f"[data] train={len(train_rows)} dev={len(dev_rows)}")

    proc, tok = get_proc()
    model = Qwen3_5ForSequenceClassification.from_pretrained(
        _MODEL, num_labels=1, dtype=torch.bfloat16
    )
    tc = model.config.get_text_config()
    tc.pad_token_id = tok.pad_token_id
    tc.eos_token_id = tok.eos_token_id
    model.num_labels = 1
    model.to(device)
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)
    if args.resume:
        model.load_adapter(args.resume, adapter_name="default")
    # 关键：混合线性注意力架构必须用非 reentrant checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    # from_pretrained may leave modules in eval mode; LoRA dropout must train.
    model.train()

    head_params = [p for n, p in model.named_parameters() if "score" in n and p.requires_grad]
    head_ids = {id(p) for p in head_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr_lora},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )
    warmup_steps = max(1, int(args.steps * args.warmup))

    def lr_at(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(max(p, 0.0), 1.0)))

    evlog = open(f"{args.out}/events.jsonl", "a", encoding="utf-8")
    step = 0
    t0 = time.time()
    order = list(range(len(train_rows)))
    random.shuffle(order)
    ptr = 0
    peak = 0.0
    aborted = False

    while step < args.steps:
        if ptr + args.bs > len(order):
            random.shuffle(order)
            ptr = 0
        idxs = order[ptr : ptr + args.bs]
        ptr += args.bs
        rows = [train_rows[i] for i in idxs]

        opt.zero_grad(set_to_none=True)
        enc, counts = encode_batch(proc, rows)
        out = model(**model_inputs(to_device(enc, device)))
        logits = out.logits.float().squeeze(-1)

        # Normalize candidates within each question, then average question
        # losses. Summing the flattened batch would scale gradients with bs.
        losses = []
        offset = 0
        for row, count in zip(rows, counts):
            gold = torch.tensor(row["gold"], dtype=torch.float32, device=device)
            gold = gold / gold.sum()
            probs = probabilities(logits[offset : offset + count])
            losses.append(soft_ce(probs, gold) + BRIER_W * brier(probs, gold))
            offset += count
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss_val = loss.item()
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite gradient")
        # Step one must have a nonzero learning rate.
        scale = lr_at(step + 1)
        opt.param_groups[0]["lr"] = args.lr_lora * scale
        opt.param_groups[1]["lr"] = args.lr_head * scale
        cur, pk = mem_gb()
        peak = max(peak, pk)
        av_now = avail_gb()
        del out, logits, probs, loss, losses

        # 预防第二关：内存硬闸。两个条件互为正交，不要拿总量反推可用量，
        # 121 GB 与 128 GB 的差异会让边界值在毫厘之间反复误判。
        if av_now is not None and av_now < args.mem_min_run:
            log(
                f"[MEM-GUARD] 可用内存 {av_now:.1f}GB 低于运行下限 {args.mem_min_run}GB，"
                f"第 {step} 步主动停训并保存现场。"
            )
            aborted = True
            break
        if pk > args.mem_cap:
            log(f"[MEM-GUARD] 峰值 {pk:.2f}GB 超过 cap {args.mem_cap}GB，第 {step} 步主动停训并保存现场。")
            aborted = True
            break

        # backward() only fills gradients. Commit the update after the guard;
        # otherwise a run can finish and save completely unchanged parameters.
        opt.step()
        step += 1
        cur, pk = mem_gb()
        peak = max(peak, pk)
        av_now = avail_gb()
        # Adam creates optimizer state on the first step, so check again after
        # updating. These sampled guards cannot prevent an OOM within an op.
        if pk > args.mem_cap or (av_now is not None and av_now < args.mem_min_run):
            aborted = True
        rec = {
            "step": step,
            "loss": round(loss_val, 5),
            "mem_cur_gb": round(cur, 2),
            "mem_peak_gb": round(pk, 2),
            "avail_gb": round(av_now, 1) if av_now is not None else None,
            "lr_scale": round(scale, 4),
            "aborted": aborted,
        }
        evlog.write(json.dumps(rec) + "\n")
        evlog.flush()
        if aborted:
            log(f"[MEM-GUARD] Stopped after optimizer update {step}")
            break

        if step % 20 == 0:
            log(
                f"[train] step {step}/{args.steps} peak={pk:.2f}GB avail={av_now}GB elapsed={time.time()-t0:.0f}s"
            )

        if torch.cuda.is_available() and step % 50 == 0:
            torch.cuda.empty_cache()

    evlog.close()

    final_dir = f"{args.out}/final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    proc.save_pretrained(final_dir)
    log(f"[save] {final_dir} aborted={aborted}")

    if aborted:
        with open(f"{args.out}/aborted.json", "w") as f:
            json.dump({"step": step, "peak_gb": round(peak, 2), "avail_gb": avail_gb()}, f)
        log("[ABORT] 因内存闸门停止，跳过评估。排查后改用更小模型或更小 batch 重跑。")
        return

    # Fit only on calibration; defer reading test until the fit is saved.
    calib_rows = load_rows(f"{args.data}/calibration.jsonl")
    best_t, cal_ce = fit_temperature(run_logits(model, calib_rows, device), calib_rows)
    save_temperature(Path(final_dir) / "temperatures.json", best_t, len(calib_rows), cal_ce)
    test_rows = load_rows(f"{args.data}/test.jsonl")
    log(f"[calib] best_T={best_t:.4f} calib_soft_ce={cal_ce:.5f}")

    dev_p = infer_probs(model, dev_rows, device)
    dev_m = metrics(dev_p, dev_rows)
    dev_p_t = infer_probs(model, dev_rows, device, temperature=best_t)
    dev_m_t = metrics(dev_p_t, dev_rows)
    test_p_t = infer_probs(model, test_rows, device, temperature=best_t)
    test_m_t = metrics(test_p_t, test_rows)
    test_p_raw = infer_probs(model, test_rows, device)
    test_m_raw = metrics(test_p_raw, test_rows)

    report = {
        "config": vars(args),
        "temperature": best_t,
        "calib_soft_ce": cal_ce,
        "dev": dev_m,
        "dev_calibrated": dev_m_t,
        "test_calibrated": test_m_t,
        "test_raw": test_m_raw,
        "peak_gb": round(peak, 2),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(f"{args.out}/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("[report] " + json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        sys.argv = [sys.argv[0], "--steps", "25", "--out", "runs/smoke", "--bs", "2"]
    main()
