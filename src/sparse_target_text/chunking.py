from __future__ import annotations

import hashlib
import pandas as pd
from src.news.text_preprocessing import stable_text_hash


def load_tokenizer(cfg: dict):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Tokenizer-aware event chunking requires transformers on the server") from exc
    kwargs = {}
    if cfg["text_encoder"].get("revision"): kwargs["revision"] = cfg["text_encoder"]["revision"]
    if cfg["text_encoder"].get("hf_cache_dir"): kwargs["cache_dir"] = cfg["text_encoder"]["hf_cache_dir"]
    if bool(cfg["text_encoder"].get("local_files_only", False)): kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], **kwargs)


def tokenizer_chunk_events(events: pd.DataFrame, cfg: dict, tokenizer=None) -> pd.DataFrame:
    """Expand category summaries into chunks that cannot be silently truncated."""
    tokenizer = tokenizer or load_tokenizer(cfg)
    max_length = int(cfg["text_encoder"]["max_length"])
    payload_length = max_length - int(tokenizer.num_special_tokens_to_add(pair=False))
    if payload_length <= 0: raise ValueError("text_encoder.max_length is too small for tokenizer special tokens")
    rows = []
    for event in events.itertuples(index=False):
        token_ids = tokenizer.encode(str(event.text), add_special_tokens=False, truncation=False)
        chunks = [token_ids[i:i + payload_length] for i in range(0, len(token_ids), payload_length)] or [[]]
        for chunk_index, ids in enumerate(chunks):
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            row = event._asdict(); row["original_event_id"] = str(event.event_id)
            row["original_text_hash"] = str(event.text_hash); row["chunk_index"] = chunk_index
            row["chunk_count"] = len(chunks); row["token_count"] = len(ids); row["text"] = text
            row["word_count"] = len(text.split())
            row["text_hash"] = stable_text_hash(text)
            row["event_id"] = hashlib.sha256(f"{event.event_id}|chunk|{chunk_index}|{row['text_hash']}".encode()).hexdigest()[:24]
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) and out.token_count.max() > payload_length:
        raise AssertionError("Tokenizer chunk exceeds model payload length")
    return out
