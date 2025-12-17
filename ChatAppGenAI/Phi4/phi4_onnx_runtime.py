# phi4_onnx_runtime.py
"""
Minimal ONNX runtime loader for a local phi4-mini-instruct ONNX export.
Place this file in the same folder as your model.onnx, tokenizer files, and configuration_phi4.py.
"""

import os
import json
from typing import List, Optional, Any
import numpy as np

try:
    import onnxruntime as ort
    ONNXRT_AVAILABLE = True
except Exception:
    ONNXRT_AVAILABLE = False

# Prefer HF tokenizer if available; fallback to a simple vocab-based tokenizer.
try:
    from transformers import AutoTokenizer
    HF_TOKENIZER_AVAILABLE = True
except Exception:
    AutoTokenizer = None
    HF_TOKENIZER_AVAILABLE = False


class SimpleVocabTokenizer:
    def __init__(self, vocab_path: str, unk_token: str = "<|endoftext|>", pad_token: str = "<|endoftext|>"):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.unk_token = unk_token
        self.pad_token = pad_token
        # ensure special tokens exist
        for t in (unk_token, pad_token):
            if t not in self.token_to_id:
                nid = max(self.token_to_id.values()) + 1
                self.token_to_id[t] = nid
                self.id_to_token[nid] = t

    def encode(self, text: str) -> List[int]:
        # naive whitespace + char fallback; this is a fallback only
        tokens = text.strip().split()
        ids: List[int] = []
        for t in tokens:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
            else:
                for ch in t:
                    ids.append(self.token_to_id.get(ch, self.token_to_id[self.unk_token]))
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_token.get(i, self.unk_token) for i in ids)


def find_onnx_file(repo_dir: str) -> Optional[str]:
    # prefer model.onnx, otherwise first .onnx
    candidates = ["model.onnx", "phi4-mini-instruct-onnx.onnx"]
    for c in candidates:
        p = os.path.join(repo_dir, c)
        if os.path.exists(p):
            return p
    for fname in os.listdir(repo_dir):
        if fname.lower().endswith(".onnx"):
            return os.path.join(repo_dir, fname)
    return None


class Phi4ONNXRuntime:
    def __init__(self, repo_dir: str, onnx_path: Optional[str] = None, device: str = "cpu"):
        self.repo_dir = repo_dir
        self.device = device
        self.onnx_path = onnx_path or find_onnx_file(repo_dir)
        if not self.onnx_path:
            raise FileNotFoundError("No ONNX model file found in repo_dir.")
        if not ONNXRT_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed. Install onnxruntime or onnxruntime‑cpu.")
        self.session = self._create_session(self.onnx_path)
        self.tokenizer = self._load_tokenizer(repo_dir)
        # try to read config.json for eos token id
        cfg_path = os.path.join(repo_dir, "config.json")
        self.config = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except Exception:
                self.config = {}

    def _create_session(self, onnx_path: str) -> ort.InferenceSession:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, (os.cpu_count() or 1) // 2)
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)

    def _load_tokenizer(self, repo_dir: str):
        # prefer HF tokenizer if present
        if HF_TOKENIZER_AVAILABLE:
            try:
                return AutoTokenizer.from_pretrained(repo_dir, use_fast=True)
            except Exception:
                pass
        vocab_path = os.path.join(repo_dir, "vocab.json")
        if os.path.exists(vocab_path):
            return SimpleVocabTokenizer(vocab_path)
        raise FileNotFoundError("No tokenizer found. Place vocab.json or a HuggingFace tokenizer in repo_dir.")

    def _infer_input_names(self):
        return [inp.name for inp in self.session.get_inputs()]

    def _prepare_feed(self, input_ids: List[int]) -> dict:
        arr = np.array([input_ids], dtype=np.int64)
        attention = np.ones_like(arr, dtype=np.int64)
        feed = {}
        for inp in self.session.get_inputs():
            name = inp.name.lower()
            if "input_ids" in name or name in ("input", "input__0"):
                feed[inp.name] = arr
            elif "attention" in name or "attention_mask" in name:
                feed[inp.name] = attention
            elif "position" in name:
                # create simple position ids if required
                pos = np.arange(arr.shape[1], dtype=np.int64)[None, :]
                feed[inp.name] = pos
            else:
                # skip other inputs (past_key_values handling not implemented here)
                pass
        return feed

    def generate(self, prompt: str, max_new_tokens: int = 64, stop_token_id: Optional[int] = None) -> str:
        # encode
        if hasattr(self.tokenizer, "encode"):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"].tolist()[0]

        generated = list(input_ids)
        eos_id = stop_token_id or self.config.get("eos_token_id")

        for _ in range(max_new_tokens):
            feed = self._prepare_feed(generated)
            outputs = self.session.run(None, feed)
            # assume logits are last output and shape [1, seq_len, vocab_size]
            logits = outputs[-1]
            if logits.ndim == 3:
                next_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                # sometimes ONNX returns [seq_len, vocab] for single batch
                next_logits = logits[-1, :]
            else:
                raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
            next_id = int(np.argmax(next_logits))
            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

        # decode
        if hasattr(self.tokenizer, "decode"):
            # if HF tokenizer, pass skip_special_tokens
            try:
                return self.tokenizer.decode(generated, skip_special_tokens=True)
            except TypeError:
                return self.tokenizer.decode(generated)
        else:
            return self.tokenizer.decode(generated)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run phi4 ONNX model locally (greedy generation)")
    parser.add_argument("repo_dir", help="Path to folder with ONNX model and tokenizer")
    parser.add_argument("--prompt", default="Hello world", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    runtime = Phi4ONNXRuntime(args.repo_dir)
    out = runtime.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print(out)