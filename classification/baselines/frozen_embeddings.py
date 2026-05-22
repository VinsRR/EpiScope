"""
Frozen transformer embeddings + shallow classifier baseline.

Encodes each paper into a single dense vector using a pretrained transformer
(SciBERT, BiomedBERT, etc.) in eval mode with no gradients, then trains a
classical multi-label classifier (logreg or linear SVM) on top with cross-
validation. The classifier-side machinery (folds, label selection, threshold,
output format) is inherited from `SupervisedCVBaseline`; only the feature
extraction step is overridden.

This is intentionally NOT fine-tuning. The transformer weights are never
updated. The point is to provide a strong-but-cheap baseline that fits the
small-data regime (~200 labeled papers) where full fine-tuning would overfit.

Two input variants are supported, controlled by `text_source`:

- "metadata":   uses "{title} [SEP] {abstract}" per paper. One forward pass
                per paper, truncated at `max_length` tokens.
- "full_text":  uses the paper's full concatenated text (the same content the
                other supervised baselines see). The text is tokenized once
                and split into non-overlapping `max_length`-token windows.
                Each window is encoded independently and the resulting per-
                window vectors are mean-pooled into a single paper-level
                vector. This avoids silent truncation of long papers.

Embeddings are cached on disk by (model, pooling, max_length, chunk-mode,
sha256(text)) so repeated CV runs over the same corpus don't recompute them.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from classification.baselines.common import (
    metadata_from_mapping,
)
from classification.baselines.supervised import SupervisedCVBaseline


#
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# allenai/scibert_scivocab_uncased
DEFAULT_FROZEN_MODEL = "allenai/scibert_scivocab_uncased"
DEFAULT_FROZEN_CACHE_DIR = "outputs/.frozen_embedding_cache"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


@dataclass
class FrozenTransformerEmbedder:
    """Frozen HuggingFace encoder producing one vector per input text.

    Parameters
    ----------
    model_name:
        HF model id (e.g. ``allenai/scibert_scivocab_uncased``,
        ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext``).
    pooling:
        ``"mean"`` (default) — mean over non-padding tokens of the last
        hidden state. ``"cls"`` — the first-token hidden state.
    max_length:
        Per-window token budget. Default 512 (the standard BERT limit).
    batch_size:
        Number of texts (or chunks) per forward pass.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` to auto-detect.
    cache_dir:
        Directory where individual embedding ``.npy`` files are stored. Pass
        ``None`` to disable disk caching (in-memory cache still applies).
    """

    model_name: str = DEFAULT_FROZEN_MODEL
    pooling: str = "mean"
    max_length: int = 512
    batch_size: int = 8
    device: Optional[str] = None
    cache_dir: Optional[str] = DEFAULT_FROZEN_CACHE_DIR

    def __post_init__(self) -> None:
        if self.pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        self._tokenizer = None
        self._model = None
        self._resolved_device: Optional[str] = None
        self._memory_cache: dict[str, np.ndarray] = {}
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    # -- model lazy-load ----------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self.device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self._resolved_device)

    # -- cache keying -------------------------------------------------------

    def _cache_key(self, text: str, *, chunked: bool) -> str:
        payload = f"{self.model_name}|{self.pooling}|{self.max_length}|{int(chunked)}|{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _disk_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return Path(self.cache_dir) / f"{key}.npy"

    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        if key in self._memory_cache:
            return self._memory_cache[key]
        path = self._disk_path(key)
        if path is not None and path.exists():
            vec = np.load(path)
            self._memory_cache[key] = vec
            return vec
        return None

    def _save_to_cache(self, key: str, vec: np.ndarray) -> None:
        self._memory_cache[key] = vec
        path = self._disk_path(key)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, vec)
            os.replace(tmp, path)

    # -- pooling ------------------------------------------------------------

    def _pool(self, last_hidden_state, attention_mask) -> "np.ndarray":
        import torch

        if self.pooling == "cls":
            pooled = last_hidden_state[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        return pooled.detach().cpu().numpy()

    # -- single-window encoding (truncation at max_length) ------------------

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._load()
        inputs = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._resolved_device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        return self._pool(outputs.last_hidden_state, inputs["attention_mask"])

    # -- multi-window encoding (chunked, then mean across windows) ----------

    def _encode_long(self, text: str) -> np.ndarray:
        import torch

        self._load()
        # Tokenize once without truncation; the tokenizer adds [CLS]/[SEP] later
        # when we re-encode each window, so we work with token *ids* here.
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = self._tokenizer.encode(self._tokenizer.unk_token or "[UNK]", add_special_tokens=False)

        # Reserve 2 special tokens per window for [CLS] and [SEP] (BERT-style).
        window_size = max(1, self.max_length - 2)
        chunks = [ids[i : i + window_size] for i in range(0, len(ids), window_size)] or [ids[:window_size]]

        # Encode windows in batches.
        per_window: list[np.ndarray] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            decoded = [self._tokenizer.decode(window, skip_special_tokens=True) for window in batch]
            inputs = self._tokenizer(
                decoded,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._resolved_device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            per_window.append(self._pool(outputs.last_hidden_state, inputs["attention_mask"]))
        stacked = np.vstack(per_window) if per_window else np.zeros((1, 0), dtype=np.float32)
        return stacked.mean(axis=0)

    # -- public encode API --------------------------------------------------

    def encode(self, texts: Sequence[str], *, chunk_long_text: bool = False) -> np.ndarray:
        """Return one vector per input text, with caching.

        With ``chunk_long_text=True``, each text is split into max_length
        token windows; the per-window embeddings are mean-pooled into one
        per-text vector. With ``chunk_long_text=False`` (default), texts
        longer than ``max_length`` tokens are silently truncated by the
        tokenizer — appropriate only for short metadata-only inputs.
        """
        texts = [t if t else " " for t in texts]
        out: list[Optional[np.ndarray]] = [None] * len(texts)
        pending: list[tuple[int, str, str]] = []  # (idx, text, cache_key)

        for idx, text in enumerate(texts):
            key = self._cache_key(text, chunked=chunk_long_text)
            cached = self._load_from_cache(key)
            if cached is not None:
                out[idx] = cached
            else:
                pending.append((idx, text, key))

        if pending:
            if chunk_long_text:
                for idx, text, key in pending:
                    vec = self._encode_long(text)
                    self._save_to_cache(key, vec)
                    out[idx] = vec
            else:
                for start in range(0, len(pending), self.batch_size):
                    batch = pending[start : start + self.batch_size]
                    vectors = self._encode_batch([t for _, t, _ in batch])
                    for (idx, _, key), vec in zip(batch, vectors):
                        self._save_to_cache(key, vec)
                        out[idx] = vec

        return np.vstack([np.asarray(v, dtype=np.float32) for v in out])


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def _frozen_corpus(
    records: Sequence[Mapping[str, Any]],
    *,
    text_source: str,
) -> list[str]:
    """Build the per-paper text strings the encoder will consume."""
    texts: list[str] = []
    for record in records:
        if text_source == "metadata":
            metadata = metadata_from_mapping(record)
            title = (metadata.title or "").strip()
            abstract = (metadata.abstract or "").strip()
            if title and abstract:
                text = f"{title} [SEP] {abstract}"
            else:
                text = title or abstract
        elif text_source == "full_text":
            text = str(record.get("_metadata_text") or "").strip()
        else:
            raise ValueError(
                f"Unknown text_source={text_source!r}; expected 'metadata' or 'full_text'."
            )
        if not text:
            text = str(record.get("paper_id") or "empty document")
        texts.append(text)
    return texts


class SupervisedFrozenEmbeddingBaseline(SupervisedCVBaseline):
    """Cross-validated supervised baseline over frozen transformer embeddings.

    Uses a frozen pretrained encoder (no fine-tuning) to produce one paper-
    level vector per paper, then trains a shallow per-label classifier
    (logreg or linear SVM) inside each CV fold. The encoder is shared across
    folds; only the classifier is refit per fold.

    Optionally fits a `StandardScaler` per fold (train-only, never on test).
    """

    def __init__(
        self,
        *,
        embedder: FrozenTransformerEmbedder,
        text_source: str = "metadata",
        normalize_features: bool = True,
        **kwargs: Any,
    ) -> None:
        if text_source not in ("metadata", "full_text"):
            raise ValueError(
                f"text_source must be 'metadata' or 'full_text', got {text_source!r}."
            )
        self._embedder = embedder
        self._text_source = text_source
        self._normalize_features = normalize_features
        super().__init__(**kwargs)

    def _build_corpus(self) -> list[str]:
        corpus = _frozen_corpus(self.records, text_source=self._text_source)
        # Pre-encode the full corpus once. This populates the embedder's disk
        # and in-memory caches, so the per-fold `_features` calls below are
        # cheap cache lookups rather than fresh transformer forward passes.
        self._embedder.encode(corpus, chunk_long_text=self._text_source == "full_text")
        return corpus

    def _features(self, train_texts: Sequence[str], test_texts: Sequence[str]):
        chunk_long = self._text_source == "full_text"
        x_train = self._embedder.encode(list(train_texts), chunk_long_text=chunk_long)
        x_test = self._embedder.encode(list(test_texts), chunk_long_text=chunk_long)

        if self._normalize_features:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        return x_train, x_test
