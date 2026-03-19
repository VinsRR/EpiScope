from __future__ import annotations
from typing import Iterable, List, Dict, Any
from tqdm import tqdm
import torch

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, AutoModel

from .base import Embedder



class HuggingFaceEmbedder(Embedder):
    """Wrapper around a HuggingFace embedding model."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 8) -> None:
        self._model = model
        self._batch_size = batch_size
        # pass model_name to the underlying class
        self._embedder = HuggingFaceEmbedding(model_name=model)
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        # if you want a true hidden size: uncomment next line
        # self._dim = cfg.hidden_size
        # but as fallback, we embed a dummy text
        self._dim = len(self.embed_text("test"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return the dense vector."""
        return self._embedder.get_text_embedding(text)  # or _get_query_embedding if your version uses that

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed an iterable of text strings in batches."""
        texts_list = list(texts)
        vectors: List[List[float]] = []
        for i in tqdm(range(0, len(texts_list), self._batch_size), desc="Embedding texts"):
            batch = texts_list[i : i + self._batch_size]
            vectors.extend(self._embedder.get_text_embedding_batch(batch))
        return vectors



class HuggingFaceSparseEmbedder:
    """Generic sparse embedder wrapper (e.g. SPLADE-like models)."""

    def __init__(self, model: str, batch_size: int = 8, device: str = "cpu", top_k: int | None = None) -> None:
        self._model = model
        self._batch_size = batch_size
        self._device = device
        self._top_k = top_k

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._encoder = AutoModelForMaskedLM.from_pretrained(model, trust_remote_code=True).to(device)
        self._encoder.eval()

    @property
    def model_name(self) -> str:
        return self._model

    def _encode_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        encoded = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._encoder(**encoded).logits  # [B, T, V]

            # SPLADE-style pooling
            weights = torch.log1p(torch.relu(logits))
            pooled = weights.max(dim=1).values  # [B, V]

        outputs = []
        for row in pooled:
            if self._top_k is not None and self._top_k < row.numel():
                vals, idx = torch.topk(row, self._top_k)
                mask = vals > 0
                indices = idx[mask].cpu().tolist()
                values = vals[mask].cpu().tolist()
            else:
                nz = torch.nonzero(row > 0, as_tuple=True)[0]
                indices = nz.cpu().tolist()
                values = row[nz].cpu().tolist()

            outputs.append({
                "indices": indices,
                "values": values,
            })
        return outputs

    def embed_text(self, text: str) -> Dict[str, Any]:
        return self._encode_batch([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        texts_list = list(texts)
        outputs: List[Dict[str, Any]] = []
        for i in tqdm(range(0, len(texts_list), self._batch_size), desc="Embedding sparse texts"):
            batch = texts_list[i:i + self._batch_size]
            outputs.extend(self._encode_batch(batch))
        return outputs
    





class HuggingFaceLateEmbedder:
    """Generic late-interaction embedder returning one vector per token."""

    def __init__(self, model: str, batch_size: int = 4, device: str = "cpu", max_length: int = 512) -> None:
        self._model = model
        self._batch_size = batch_size
        self._device = device
        self._max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._encoder = AutoModel.from_pretrained(model, trust_remote_code=True).to(device)
        self._encoder.eval()

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        self._dim = cfg.hidden_size

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_batch(self, batch: List[str]) -> List[List[List[float]]]:
        encoded = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._encoder(**encoded)
            token_embeddings = outputs.last_hidden_state  # [B, T, D]

        attention_mask = encoded["attention_mask"]
        batch_outputs: List[List[List[float]]] = []

        for emb, mask in zip(token_embeddings, attention_mask):
            valid_len = int(mask.sum().item())
            token_vecs = emb[:valid_len]

            # Optional: remove CLS/SEP if desired
            # token_vecs = token_vecs[1:-1]

            # Optional normalization for late interaction
            token_vecs = torch.nn.functional.normalize(token_vecs, p=2, dim=1)

            batch_outputs.append(token_vecs.cpu().tolist())

        return batch_outputs

    def embed_text(self, text: str) -> List[List[float]]:
        return self._encode_batch([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[List[float]]]:
        texts_list = list(texts)
        outputs: List[List[List[float]]] = []
        for i in tqdm(range(0, len(texts_list), self._batch_size), desc="Embedding late-interaction texts"):
            batch = texts_list[i:i + self._batch_size]
            outputs.extend(self._encode_batch(batch))
        return outputs