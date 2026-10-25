"""起推理 HTTP 服务。

输入一条请求：图像、上下文、判别问题、候选列表。
输出每个候选的校准后概率与最优候选。

刻意只用标准库和 transformers，不加 Web 框架，减少读者的环境负担。
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration import probabilities, load_temperature
from inputs import encode_batch, model_inputs


def build_engine(ckpt, model, temperature):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForSequenceClassification

    proc = AutoProcessor.from_pretrained(ckpt)
    base = Qwen3_5ForSequenceClassification.from_pretrained(
        model, num_labels=1, dtype=torch.bfloat16
    )
    tc = base.config.get_text_config()
    tok = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tc.pad_token_id = tok.pad_token_id
    tc.eos_token_id = tok.eos_token_id
    base.num_labels = 1

    from peft import PeftModel

    net = PeftModel.from_pretrained(base, ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device).eval()
    return proc, tok, net, device, temperature, torch


def decide(engine, payload):
    proc, tok, net, device, temperature, torch = engine
    enc, _ = encode_batch(proc, [payload])
    enc = model_inputs(enc)
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    with torch.no_grad():
        logits = net(**enc).logits.float().squeeze(-1)
    probs = probabilities(logits, temperature).cpu().tolist()
    best = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "options": payload["options"],
        "probs": probs,
        "prediction": best,
        "confidence": probs[best],
        "temperature": temperature,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LoRA adapter 目录，如 runs/v2/final")
    ap.add_argument("--model", required=True, help="底座本地路径，如 models/Qwen3.5-2B")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--temperatures", help="Calibration JSON exported by evaluate.py")
    args = ap.parse_args()

    temp = load_temperature(args.ckpt, args.temperature, args.temperatures)

    engine = build_engine(args.ckpt, args.model, temp)
    print(f"[serve] ckpt={args.ckpt} T={temp} port={args.port}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                out = decide(engine, payload)
                body = json.dumps(out, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": repr(e)}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    HTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
