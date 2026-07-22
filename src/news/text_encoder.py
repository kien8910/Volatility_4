from __future__ import annotations

import hashlib

import numpy as np
import torch


class HashingTextEncoder:
    """Deterministic local encoder used for tests and schema smoke checks."""

    def __init__(self, embedding_dim: int = 16, model_name: str = "hashing-test", max_length: int = 256):
        self.embedding_dim = int(embedding_dim)
        self.model_name = model_name
        self.max_length = int(max_length)

    def encode(self, texts: list[str], pooling_method: str = "cls") -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(f"{pooling_method}|{text}".encode("utf-8")).digest()
            raw = np.frombuffer((digest * ((self.embedding_dim // len(digest)) + 1))[: self.embedding_dim], dtype=np.uint8)
            vec = (raw.astype(np.float32) / 127.5) - 1.0
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm > 0 else vec)
        return np.vstack(rows).astype(np.float32)


class FrozenFinancialTextEncoder:
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length: int = 256,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        use_amp: bool = False,
        normalize_embeddings: bool = False,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Step 6 text encoding requires transformers. Install it on the server, "
                "or set text_encoder.model_name='hashing-test' for local smoke tests."
            ) from exc
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.use_amp = bool(use_amp)
        self.normalize_embeddings = bool(normalize_embeddings)
        load_kwargs = {"local_files_only": bool(local_files_only)}
        if revision: load_kwargs["revision"] = revision
        if cache_dir: load_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None:
            raise ValueError(f"Cannot infer embedding dimension for encoder {model_name}")
        self.embedding_dim = int(hidden)

    def encode(self, texts: list[str], pooling_method: str = "cls") -> np.ndarray:
        if pooling_method not in {"cls", "mean"}:
            raise ValueError("pooling_method must be 'cls' or 'mean'")
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp and self.device.type == "cuda"):
                    model_out = self.model(**encoded)
                    last_hidden = model_out.last_hidden_state
                    if pooling_method == "cls":
                        pooled = last_hidden[:, 0, :]
                    else:
                        mask = encoded["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
                        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    if self.normalize_embeddings:
                        pooled = torch.nn.functional.normalize(pooled, dim=-1)
                outputs.append(pooled.detach().float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


def build_text_encoder(cfg: dict, device: str | torch.device):
    model_name = str(cfg["text_encoder"]["model_name"])
    if model_name == "hashing-test":
        return HashingTextEncoder(
            embedding_dim=int(cfg["text_encoder"].get("hashing_dim", 16)),
            model_name=model_name,
            max_length=int(cfg["text_encoder"].get("max_length", 256)),
        )
    return FrozenFinancialTextEncoder(
        model_name=model_name,
        max_length=int(cfg["text_encoder"]["max_length"]),
        batch_size=int(cfg["text_encoder"]["batch_size"]),
        device=device,
        use_amp=bool(cfg["runtime"].get("use_amp", False)),
        normalize_embeddings=bool(cfg["text_encoder"].get("normalize_embeddings", False)),
        revision=cfg["text_encoder"].get("revision"),
        cache_dir=cfg["text_encoder"].get("hf_cache_dir"),
        local_files_only=bool(cfg["text_encoder"].get("local_files_only", False)),
    )
