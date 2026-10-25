"""Check the local .venv against the text training requirements and CUDA needs.

Run ``python test_env.py``; the script switches to .venv automatically.
Use ``--device 1`` to check another logical CUDA device. CUDA_VISIBLE_DEVICES
still applies, just as it does when running the training script.

Only small synthetic tensors and a randomly initialized tiny Qwen3 are used.
No model/data downloads or full training runs are performed. Exit code 0 means
all checks passed; exit code 1 means the environment is not ready.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "text_jev_train" / "requirements.txt"


class Checks:
    """Continue after individual failures so one run gives a useful diagnosis."""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def run(self, label, action):
        try:
            detail = action()
        except Exception as exc:
            self.failed += 1
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}", flush=True)
            return False
        self.passed += 1
        print(f"[PASS] {label}: {detail or 'OK'}", flush=True)
        return True


def check_dependencies(checks):
    # Distribution names can differ from their Python import names.
    from packaging.requirements import Requirement

    import_names = {"huggingface-hub": "huggingface_hub"}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue

        def check_one(requirement=requirement):
            version = importlib.metadata.version(requirement.name)
            if not requirement.specifier.contains(version, prereleases=True):
                raise RuntimeError(f"installed {version}, required {requirement}")
            name = requirement.name.lower().replace("_", "-")
            importlib.import_module(import_names.get(name, name.replace("-", "_")))
            return f"{version}; version and import OK"

        checks.run(str(requirement), check_one)
    return "requirements file processed"


def check_qwen3():
    # Satisfying the declared transformers minimum alone does not guarantee
    # Qwen3 support. Resolve the actual lazy imports used by text training.
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

    return f"{AutoTokenizer.__name__}, {Qwen3Config.__name__}, {Qwen3Model.__name__}"


def show_driver():
    # nvidia-smi reports the driver's CUDA ceiling, NOT torch's CUDA runtime.
    # Its absence is advisory: the actual CUDA operations below are decisive.
    command = shutil.which("nvidia-smi")
    if command is None:
        print("[WARN] nvidia-smi unavailable; continuing with PyTorch checks")
        return
    try:
        result = subprocess.run(
            [command, "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        print("Physical GPUs / driver (indices may differ from logical CUDA indices):")
        print(result.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[WARN] nvidia-smi failed: {exc}")


def check_cuda(device_index):
    import torch

    print(f"PyTorch: {torch.__version__}; bundled CUDA: {torch.version.cuda}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if torch.version.cuda is None:
        raise RuntimeError("CPU-only PyTorch build; install a CUDA-enabled build")
    # Preserve PyTorch's driver diagnosis without emitting a local source path.
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always")
        available = torch.cuda.is_available()
    for diagnostic in diagnostics:
        print(f"[WARN] {diagnostic.message}")
    if not available:
        raise RuntimeError("CUDA initialization failed; check driver and GPU visibility")
    count = torch.cuda.device_count()
    if not 0 <= device_index < count:
        raise RuntimeError(f"cuda:{device_index} is outside the {count} visible devices")
    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    free, total = torch.cuda.mem_get_info(device_index)
    print(f"Visible CUDA devices: {count}; build architectures: {torch.cuda.get_arch_list()}")
    print(f"Selected GPU memory: {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
    return (
        f"cuda:{device_index}: {properties.name}; "
        f"compute capability {properties.major}.{properties.minor}"
    )


def check_fp32(device):
    import numpy as np
    import torch

    # Also exercise the NumPy bridge used when preparing numeric inputs.
    x = torch.from_numpy(np.ones((32, 32), dtype=np.float32)).to(device)
    result = x @ x
    torch.cuda.synchronize(device)  # Surface asynchronous kernel/driver errors here.
    if not torch.allclose(result, torch.full_like(result, 32.0)):
        raise RuntimeError("FP32 matrix multiplication returned incorrect values")
    return "NumPy bridge, GPU allocation, FP32 matmul and synchronization"


def check_bf16_training(device):
    import torch
    from transformers import Qwen3Config, Qwen3Model

    # Training explicitly uses BF16 autocast. CPU fallback or BF16 emulation
    # alone is insufficient to declare this GPU environment ready.
    major, _ = torch.cuda.get_device_capability(device)
    if major < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 support is required by text training")

    # Construct locally from config: never call from_pretrained or access the Hub.
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        use_cache=False,
    )
    backbone = Qwen3Model(config).to(device).train()
    head = torch.nn.Linear(64, 8).to(device).train()
    parameters = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    tokens = torch.randint(0, config.vocab_size, (2, 16), device=device)
    targets = torch.tensor([0, 1], device=device)
    before = head.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = backbone(input_ids=tokens, attention_mask=torch.ones_like(tokens))
        logits = head(hidden.last_hidden_state[:, -1])
        loss = torch.nn.functional.cross_entropy(logits.float(), targets)
    if logits.dtype != torch.bfloat16 or not torch.isfinite(loss).item():
        raise RuntimeError("BF16 autocast failed or loss is non-finite")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in parameters):
        raise RuntimeError("backward produced missing or non-finite gradients")
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    if any(not torch.isfinite(p).all().item() for p in parameters):
        raise RuntimeError("AdamW produced non-finite parameters")
    if torch.equal(before, head.weight.detach()):
        raise RuntimeError("AdamW did not update the classifier weights")
    return f"tiny Qwen3 BF16 forward/backward, gradient clipping, AdamW; loss={loss.item():.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="logical CUDA index (default: 0)")
    args = parser.parse_args()

    # Compare prefixes, not resolved executables: venv Python is often a symlink
    # to an interpreter outside the venv. This also works without activation.
    if Path(sys.prefix).resolve() != VENV.resolve():
        python = VENV / "bin" / "python"
        if not python.is_file():
            print("[FAIL] .venv/bin/python is missing or its symlink is broken")
            return 1
        print("Switching to .venv/bin/python", flush=True)
        try:
            return subprocess.run(
                [str(python), str(Path(__file__).resolve()), *sys.argv[1:]]
            ).returncode
        except OSError as exc:
            print(f"[FAIL] Cannot start .venv/bin/python: {exc}")
            return 1

    print(f"Environment: .venv; Python {sys.version.split()[0]}", flush=True)
    checks = Checks()
    checks.run("text_jev_train/requirements.txt", lambda: check_dependencies(checks))
    qwen_ready = checks.run("Text training API imports", check_qwen3)
    show_driver()
    cuda_ready = checks.run("CUDA runtime and selected GPU", lambda: check_cuda(args.device))
    if cuda_ready:
        device = f"cuda:{args.device}"
        checks.run("FP32 GPU computation", lambda: check_fp32(device))
        if qwen_ready:
            checks.run("BF16 training smoke check", lambda: check_bf16_training(device))
        else:
            print("[SKIP] BF16 training smoke check: Qwen3 imports failed")
    else:
        print("[SKIP] GPU computations: CUDA unavailable or invalid device")
    print(f"\nResult: {checks.passed} passed, {checks.failed} failed")
    if checks.failed:
        print("NOT READY: fix the failed checks before running text training.")
        return 1
    print("READY: dependency and GPU smoke checks passed.")
    print("Full training memory requirements and local model/data files are not checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
