"""CPU regressions for the published pipelines; no model/data downloads."""
import ast
import base64
from contextlib import nullcontext
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision_jev_train" / "scripts"
sys.path.insert(0, str(VISION))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vision_train = load_module("vision_train", VISION / "train.py")
vision_eval = load_module("vision_eval", VISION / "evaluate.py")
vision_serve = load_module("vision_serve", VISION / "serve.py")
prepare = load_module("vision_prepare", VISION / "prepare_data.py")
text_train = load_module("text_train", ROOT / "text_jev_train/scripts/train.py")
import calibration
import inputs


class TinyModel(torch.nn.Module):
    """Replace only the pretrained network, keeping real autograd and AdamW."""

    def __init__(self, weight=0.0):
        super().__init__()
        self.score = torch.nn.Linear(1, 1, bias=False)
        self.score.weight.data.fill_(weight)
        self.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace())
        self.calls = 0
        self.modes = []

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        self.modes.append(self.training)
        return SimpleNamespace(logits=self.score(input_ids.float()))

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def print_trainable_parameters(self):
        pass

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(directory) / "tiny.pt")


class TinyProcessor:
    tokenizer = SimpleNamespace(pad_token="pad", pad_token_id=0, eos_token_id=1)

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


def fake_encoding(proc, rows):
    return {"input_ids": torch.tensor([[1.0], [-1.0]] * len(rows))}, [2] * len(rows)


def fake_backends(monkeypatch, model, proc):
    transformers = ModuleType("transformers")
    transformers.Qwen3_5ForSequenceClassification = SimpleNamespace(
        from_pretrained=lambda *a, **kw: model
    )
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda *a, **kw: proc)
    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kw: kw
    peft.get_peft_model = lambda base, config: base
    peft.PeftModel = SimpleNamespace(from_pretrained=lambda base, ckpt: base)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def row(gold=None):
    return {
        "input_schema": inputs.INPUT_SCHEMA,
        "options": ["yes", "no"],
        "gold": gold or [1.0, 0.0],
        "state": "context",
        "question": "question",
    }


def write_splits(directory, gold=None):
    directory.mkdir()
    for name in ("train", "dev", "calibration", "test"):
        (directory / f"{name}.jsonl").write_text(json.dumps(row(gold)) + "\n")


@pytest.mark.parametrize("guard_after_update", [False, True])
def test_vision_training_updates_parameters_and_checks_optimizer_memory(
    tmp_path, monkeypatch, guard_after_update
):
    model, proc = TinyModel().eval(), TinyProcessor()
    fake_backends(monkeypatch, model, proc)
    monkeypatch.setattr(vision_train, "get_proc", lambda: (proc, proc.tokenizer))
    monkeypatch.setattr(vision_train, "encode_batch", fake_encoding)
    monkeypatch.setattr(vision_train, "avail_gb", lambda: 100.0)
    memory = iter([(0, 0), (80, 80)]) if guard_after_update else None
    monkeypatch.setattr(vision_train, "mem_gb", lambda: next(memory) if memory else (0, 0))
    data, out = tmp_path / "prepared", tmp_path / "run"
    write_splits(data)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--model",
            "unused",
            "--data",
            str(data),
            "--out",
            str(out),
            "--steps",
            "1",
            "--bs",
            "1",
        ],
    )
    with pytest.warns(
        UserWarning, match="grid boundary"
    ) if not guard_after_update else nullcontext():
        vision_train.main()
    assert model.modes[0] is True
    state = torch.load(out / "final/tiny.pt", weights_only=True)
    assert state["score.weight"].item() > 0, "backward without step leaves the weight unchanged"
    assert (out / "aborted.json").exists() == guard_after_update
    assert (out / "final/temperatures.json").exists() != guard_after_update
    assert (out / "report.json").exists() != guard_after_update
    event = json.loads((out / "events.jsonl").read_text())
    assert event["step"] == 1 and event["lr_scale"] > 0


def test_eval_and_service_apply_one_temperature_and_reuse_raw_logits(tmp_path, monkeypatch):
    model, proc = TinyModel(1.0), TinyProcessor()
    fake_backends(monkeypatch, model, proc)
    monkeypatch.setattr(vision_eval, "encode_batch", fake_encoding)
    monkeypatch.setattr(vision_serve, "encode_batch", fake_encoding)
    data, out = tmp_path / "prepared", tmp_path / "evaluation"
    write_splits(data, [0.75, 0.25])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--model",
            "unused",
            "--ckpt",
            "unused",
            "--data",
            str(data),
            "--out",
            str(out),
        ],
    )
    vision_eval.main()
    assert model.calls == 3, "dev, calibration and test should each be encoded once"
    report = json.loads((out / "eval.json").read_text())
    temperature = calibration.load_temperature("unused", path=out / "temperatures.json")
    assert temperature != 1
    expected = torch.softmax(torch.tensor([1.0, -1.0]) / temperature, -1)
    result = vision_serve.decide((proc, proc.tokenizer, model, "cpu", temperature, torch), row())
    torch.testing.assert_close(torch.tensor(result["probs"]), expected)
    expected_ce = -(torch.tensor([0.75, 0.25]) * expected.log()).sum().item()
    assert report["splits"]["test"]["calibrated"]["soft_ce"] == pytest.approx(expected_ce)
    assert not torch.allclose(
        expected, torch.softmax(torch.tensor([1.0, -1.0]) / temperature**2, -1)
    )


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_temperatures_are_rejected(value):
    with pytest.raises(ValueError):
        calibration.probabilities(torch.tensor([1.0, 0.0]), value)


def test_missing_calibration_is_not_silently_disabled(tmp_path):
    with pytest.raises(FileNotFoundError):
        calibration.load_temperature(tmp_path)
    assert calibration.load_temperature(tmp_path, override=1) == 1


def science_row(index):
    image = Image.new("RGB", (2, 2), (index, 0, 0))
    return {
        "image": image,
        "choices": ["a", "b"],
        "answer": 0,
        "lecture": "PRIVATE EXPLANATION",
        "solution": "SECRET ANSWER",
        "hint": "public hint",
        "question": "which?",
    }


def test_answer_annotations_excluded_and_split_order_independent():
    rows = [science_row(i) for i in range(20)]
    rows.append(science_row(0))
    a, skipped = prepare.split_rows(rows)
    b, _ = prepare.split_rows(list(reversed(rows)))
    assert a == b and skipped["duplicate_image"] == 1
    assert {k: len(v) for k, v in a.items()} == {"train": 12, "dev": 2, "calibration": 2, "test": 4}
    images = [r["image_sha256"] for split in a.values() for r in split]
    assert len(images) == len(set(images)) == 20
    assert all(
        "SECRET" not in r["state"] and "PRIVATE" not in r["state"]
        for split in a.values()
        for r in split
    )
    changed = science_row(0)
    changed.update(answer=1, lecture="different", solution="different")
    assert prepare.convert(changed)["state"] == prepare.convert(science_row(0))["state"]
    with pytest.raises(ValueError, match="four"):
        prepare.split_rows(rows[:3])


def test_old_prepared_data_is_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"state": "old answer explanation"}))
    with pytest.raises(ValueError, match="Regenerate"):
        inputs.load_rows(path)


def test_all_vision_paths_share_preprocessing():
    assert vision_train.encode_batch is vision_eval.encode_batch is vision_serve.encode_batch
    image = io.BytesIO()
    Image.new("RGB", (640, 320)).save(image, format="PNG")
    sample = row() | {"image_png_b64": base64.b64encode(image.getvalue()).decode()}
    calls = []
    encoded, counts = inputs.encode_batch(lambda **kw: calls.append(kw) or {}, [sample])
    assert counts == [2]
    assert calls[0]["images"][0].size == (320, 160)
    assert "truncation" not in calls[0]


@pytest.mark.parametrize("improves", [False, True])
def test_text_initialization_and_final_step_can_win(tmp_path, monkeypatch, improves):
    class TextModel(torch.nn.Module):
        def __init__(self, *a):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0, 0.0]))

        def to(self, *a):
            return self

        def forward(self, batch):
            return {"logits": self.weight[None, :]}

    transformer = ModuleType("transformers")
    transformer.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(pad_token_id=1, eos_token_id=2)
    )
    agent = ModuleType("agentjev.model")
    agent.AgentJevModel = TextModel
    monkeypatch.setitem(sys.modules, "transformers", transformer)
    monkeypatch.setitem(sys.modules, "agentjev.model", agent)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    example = {"paths": [[1], [2]], "target": [1.0, 0.0], "question": {"type": "choice"}}
    out = tmp_path / "run"

    def load(split, directory):
        if split == "calibration":
            assert (out / "selection.json").is_file()
        if split == "test":
            assert (out / "temperatures.json").is_file()
        return [example]

    monkeypatch.setattr(text_train, "load", load)
    monkeypatch.setattr(text_train, "tokenize", lambda rows, tok: rows)
    monkeypatch.setattr(
        text_train,
        "batch",
        lambda rows, pad: {
            "target": torch.tensor([[1.0, 0.0]]),
            "cand_mask": torch.tensor([[True, True]]),
        },
    )
    evaluations = []

    def evaluate(model, rows, pad):
        evaluations.append(model.weight.detach().clone())
        p = 0.8 if len(evaluations) == 1 else (0.9 if improves else 0.7)
        return [
            {
                "type": "choice",
                "workflow": "demo",
                "target": [1.0, 0.0],
                "logits": [1.0, 0.0],
                "probs": [p, 1 - p],
            }
        ]

    monkeypatch.setattr(text_train, "evaluate", evaluate)
    protocol = json.loads((ROOT / "text_jev_train/scripts/protocol.json").read_text())
    protocol.update(max_steps=1, microbatch_questions=1, gradient_accumulation=1)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--backbone", "unused", "--protocol", str(path), "--run-dir", str(out)],
    )
    text_train.main()
    selection = json.loads((out / "selection.json").read_text())
    assert selection["best_step"] == int(improves)
    ckpt = torch.load(out / "best.pt", weights_only=True)
    expected = evaluations[1] if improves else evaluations[0]
    torch.testing.assert_close(ckpt["state_dict"]["weight"], expected)
    torch.testing.assert_close(evaluations[2], expected)
    assert (out / "report.json").is_file()


def test_metask_temperature_path_is_independent_of_cwd(tmp_path, monkeypatch):
    path = ROOT / "metask/metask-bench.py"
    tree = ast.parse(path.read_text())
    # Execute the real configuration assignments without invoking the GPU benchmark.
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in {"TEMP_PATH", "TEMP", "BY_KIND"} for t in n.targets
        )
    ]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "temperature.json").write_text('{"by_kind": {"choice": 99}}')
    namespace = {"__file__": str(path), "Path": Path, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["BY_KIND"]["choice"] == 1.9


@pytest.mark.parametrize("fail_arm", ["", "rlcd"])
def test_ab_script_is_portable_and_stops_on_failure(tmp_path, fail_arm):
    import os

    # Stub the training command, not bash/tee: exercise quoting, directories,
    # failure propagation and summary handling with real shell processes.
    runner = tmp_path / "fake python"
    runner.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-c":
    raise SystemExit(0)
if args[0] == "-":
    sys.argv = sys.argv[1:]
    exec(sys.stdin.read())
else:
    out = Path(args[args.index("--output-dir") + 1])
    print("fake training " + out.name, flush=True)
    if out.name == os.environ.get("FAKE_FAIL_ARM"):
        raise SystemExit(7)
    out.mkdir(parents=True)
    after = {"accuracy": .75, "ece": .1, "brier": .05,
             "by_type": {k: {"accuracy": .75} for k in ("choice", "score", "noul")}}
    (out / "train_report.json").write_text(json.dumps({"after": after}))
"""
    )
    runner.chmod(0o755)
    data = tmp_path / "sample.jsonl"
    data.write_text("{}\n")
    env = dict(
        os.environ,
        JEV_PYTHON=str(runner),
        JEV_TRAIN_DATA=str(data),
        JEV_EVAL_DATA=str(data),
        JEV_RUN_ROOT=str(tmp_path / "run outputs"),
        JEV_LOG_DIR=str(tmp_path / "logs"),
        FAKE_FAIL_ARM=fail_arm,
    )
    result = subprocess.run(
        ["bash", str(ROOT / "laya_rlcd_train/scripts/run_ab_compare.sh")],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    logs = tmp_path / "logs"
    assert (logs / "rlcd.log").exists()
    if fail_arm:
        assert result.returncode != 0
        assert not (logs / "soft-ce.log").exists()
        assert not (logs / "ab-summary.log").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "n/a" in (logs / "ab-summary.log").read_text()
        assert (logs / "soft-ce.log").exists()


def test_classification_runners_share_examples_and_criteria():
    sys.path.insert(0, str(ROOT))
    import benchmark_cases

    for name in ("laya/laya-zh-en.py", "metask/metask-bench.py"):
        path = ROOT / name
        imports = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ImportFrom) and n.module == "benchmark_cases"
        ]
        namespace = {}
        exec(compile(ast.Module(body=imports, type_ignores=[]), str(path), "exec"), namespace)
        assert namespace.get("CASES", namespace.get("EN_CASES")) is benchmark_cases.CASES
        assert namespace["CRITERIA"] is benchmark_cases.CRITERIA
        assert namespace["INSTRUCTIONS"] is benchmark_cases.INSTRUCTIONS
