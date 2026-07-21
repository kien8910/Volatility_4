from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class TextEncoderProtocol(Protocol):
    embedding_dim: int

    def encode(self, texts: list[str], pooling_method: str) -> np.ndarray:
        ...


@dataclass(frozen=True)
class EmbeddingKey:
    encoder_name: str
    text_hash: str
    pooling_method: str
    max_length: int


class EmbeddingCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_path = self.cache_dir / "text_embeddings.parquet"
        self.manifest_path = self.cache_dir / "embedding_manifest.csv"

    def read(self) -> pd.DataFrame:
        if not self.embedding_path.exists() or self.embedding_path.stat().st_size == 0:
            return pd.DataFrame(
                columns=[
                    "text_hash",
                    "hierarchy",
                    "encoder_name",
                    "pooling_method",
                    "max_length",
                    "embedding_dim",
                    "embedding",
                    "missing_mask",
                    "created_at",
                ]
            )
        return pd.read_parquet(self.embedding_path)

    def write(self, frame: pd.DataFrame) -> None:
        frame = frame.copy()
        if not frame.empty:
            values = np.asarray(frame["embedding"].tolist(), dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Embedding cache contains NaN or infinite values.")
        tmp = self.embedding_path.with_suffix(".tmp.parquet")
        frame.to_parquet(tmp, index=False)
        tmp.replace(self.embedding_path)

    def get_or_encode(
        self,
        requests: pd.DataFrame,
        encoder: TextEncoderProtocol,
        encoder_name: str,
        pooling_method: str,
        max_length: int,
    ) -> pd.DataFrame:
        required = {"text_hash", "hierarchy", "text", "missing_mask"}
        missing = sorted(required - set(requests.columns))
        if missing:
            raise ValueError(f"Embedding requests missing columns: {missing}")
        existing = self.read()
        key_cols = ["encoder_name", "text_hash", "pooling_method", "max_length"]
        requests = requests.drop_duplicates(["hierarchy", "text_hash"]).copy()
        requests["encoder_name"] = encoder_name
        requests["pooling_method"] = pooling_method
        requests["max_length"] = int(max_length)
        available_keys = set()
        if not existing.empty:
            available_keys = set(map(tuple, existing[key_cols].astype(str).to_numpy()))
        to_add = []
        now = pd.Timestamp.utcnow().isoformat()
        for row in requests.itertuples(index=False):
            key = (str(row.encoder_name), str(row.text_hash), str(row.pooling_method), str(row.max_length))
            if key in available_keys:
                continue
            if int(row.missing_mask) == 1 or str(row.text) == "":
                emb = np.zeros(int(encoder.embedding_dim), dtype=np.float32)
            else:
                emb = encoder.encode([str(row.text)], pooling_method=pooling_method)[0].astype(np.float32)
            if not np.isfinite(emb).all():
                raise ValueError(f"Non-finite embedding for text_hash={row.text_hash}")
            to_add.append(
                {
                    "text_hash": str(row.text_hash),
                    "hierarchy": str(row.hierarchy),
                    "encoder_name": encoder_name,
                    "pooling_method": pooling_method,
                    "max_length": int(max_length),
                    "embedding_dim": int(len(emb)),
                    "embedding": emb.tolist(),
                    "missing_mask": int(row.missing_mask),
                    "created_at": now,
                }
            )
        if to_add:
            existing = pd.concat([existing, pd.DataFrame(to_add)], ignore_index=True)
            existing = existing.drop_duplicates(key_cols, keep="last").reset_index(drop=True)
            self.write(existing)
        self._write_manifest(existing)
        return self.read()

    def _write_manifest(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            pd.DataFrame(
                columns=["encoder_name", "pooling_method", "max_length", "embedding_dim", "rows", "unique_text_hashes"]
            ).to_csv(self.manifest_path, index=False)
            return
        manifest = (
            frame.groupby(["encoder_name", "pooling_method", "max_length", "embedding_dim"], dropna=False)
            .agg(rows=("text_hash", "size"), unique_text_hashes=("text_hash", "nunique"))
            .reset_index()
        )
        manifest.to_csv(self.manifest_path, index=False)


def embedding_lookup(frame: pd.DataFrame, encoder_name: str, pooling_method: str, max_length: int) -> dict[str, np.ndarray]:
    sub = frame[
        (frame["encoder_name"].astype(str) == str(encoder_name))
        & (frame["pooling_method"].astype(str) == str(pooling_method))
        & (frame["max_length"].astype(int) == int(max_length))
    ]
    out: dict[str, np.ndarray] = {}
    for row in sub.itertuples(index=False):
        out[str(row.text_hash)] = np.asarray(json.loads(row.embedding) if isinstance(row.embedding, str) else row.embedding, dtype=np.float32)
    return out

