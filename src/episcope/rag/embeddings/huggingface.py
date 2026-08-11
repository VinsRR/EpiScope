from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List, Literal, Optional
from tqdm import tqdm
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, AutoModel

from .base import Embedder, SparseEmbedder, LateEmbedder
from episcope.rag.device import resolve_device


# SUPPORTED_CROSS_ENCODER_MODELS = {
#     "cross-encoder/ms-marco-MiniLM-L6-v2",
#     "cross-encoder/ms-marco-MiniLM-L4-v2",
# }


def _resolve_default_max_length(
    tokenizer, fallback: Optional[int] = None
) -> Optional[int]:
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is None:
        return fallback
    if isinstance(model_max_length, int) and model_max_length > 100_000:
        return fallback
    return model_max_length


SUPPORTED_DENSE_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    # bge-m3: 1024-d, 8192-token window, and (unlike bge-*-v1.5 and e5-*)
    # needs no query instruction prefix, which suits the symmetric
    # embed_query/embed_text path used here.
    "BAAI/bge-m3",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
}


class HuggingFaceEmbedder(Embedder):
    """Dense text embedder backed by SentenceTransformer."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 8,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        normalize: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        trust_remote_code: bool = True,
        prompts: Optional[Dict[str, str]] = None,
        default_prompt_name: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if model not in SUPPORTED_DENSE_MODELS:
            raise ValueError(
                f"Model '{model}' is not in the list of supported dense embedding models: "
                f"{sorted(SUPPORTED_DENSE_MODELS)}."
            )

        # Default to a safe device: auto-selecting CUDA on an unsupported GPU
        # crashes with cudaErrorNoKernelImageForDevice.
        device = resolve_device(device)

        self._model = model
        self._batch_size = batch_size
        self._normalize = normalize
        self._precision = precision
        self._truncate_dim = truncate_dim
        resolved_processor_kwargs = (
            processor_kwargs if processor_kwargs is not None else tokenizer_kwargs or {}
        )

        encoder_kwargs: Dict[str, Any] = {
            "model_name_or_path": model,
            "device": device,
            "trust_remote_code": trust_remote_code,
            "prompts": prompts,
            "default_prompt_name": default_prompt_name,
            "truncate_dim": truncate_dim,
            "model_kwargs": model_kwargs or {},
            "config_kwargs": config_kwargs or {},
        }
        sentence_transformer_params = inspect.signature(
            SentenceTransformer.__init__
        ).parameters
        if "processor_kwargs" in sentence_transformer_params:
            encoder_kwargs["processor_kwargs"] = resolved_processor_kwargs
        else:
            encoder_kwargs["tokenizer_kwargs"] = resolved_processor_kwargs

        self._encoder = SentenceTransformer(**encoder_kwargs)

        if max_length is not None:
            self._encoder.max_seq_length = max_length

        self._max_length = self._encoder.get_max_seq_length()
        self._dim = len(self.embed_text("test"))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_length(self) -> Optional[int]:
        return self._max_length

    def embed_text(self, text: str) -> List[float]:
        vector = self._encoder.encode(
            text,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            precision=self._precision,
        )
        return vector.tolist()

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        texts_list = list(texts)
        if not texts_list:
            return []
        vectors = self._encoder.encode(
            texts_list,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            precision=self._precision,
        )
        return vectors.tolist()


SUPPORTED_SPARSE_MODELS = {
    "naver/splade-cocondenser-ensembledistil",
    "naver/splade-cocondenser-selfdistil",
    "naver/splade-v3-distilbert",
}


class HuggingFaceSparseEmbedder(SparseEmbedder):
    """Sparse embedder for SPLADE-style models."""

    def __init__(
        self,
        model: str,
        batch_size: int = 8,
        device: str = "cpu",
        top_k: Optional[int] = None,
        max_length: Optional[int] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if model not in SUPPORTED_SPARSE_MODELS:
            raise ValueError(
                f"Model '{model}' is not in the list of supported sparse embedding models: "
                f"{sorted(SUPPORTED_SPARSE_MODELS)}."
            )

        self._model = model
        self._batch_size = batch_size
        self._device = device
        self._top_k = top_k

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._max_length = (
            max_length
            if max_length is not None
            else _resolve_default_max_length(self._tokenizer)
        )

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self._encoder = AutoModelForMaskedLM.from_pretrained(model, **model_kwargs).to(
            device
        )
        self._encoder.eval()

    @property
    def model_name(self) -> str:
        return self._model

    def _encode_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        encoded = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._encoder(**encoded).logits  # [B, T, V]
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
            outputs.append({"indices": indices, "values": values})

        return outputs

    def embed_text(self, text: str) -> Dict[str, Any]:
        return self._encode_batch([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        texts_list = list(texts)
        outputs: List[Dict[str, Any]] = []
        for i in tqdm(
            range(0, len(texts_list), self._batch_size), desc="Embedding sparse texts"
        ):
            outputs.extend(self._encode_batch(texts_list[i : i + self._batch_size]))
        return outputs


SUPPORTED_LATE_MODELS = {
    "colbert-ir/colbertv2.0",
    "answerdotai/answerai-colbert-small-v1",
    # "jinaai/jina-colbert-v2", # Jina models are too heavy
    # "jinaai/jina-colbert-v2-64",
}

_LATE_MODEL_PREFIXES = {
    "colbert-ir/colbertv2.0": {
        "query": "",
        "document": "",
    },
    "answerdotai/answerai-colbert-small-v1": {
        "query": "",
        "document": "",
    },
    "jinaai/jina-colbert-v2": {
        "query": "[QueryMarker]",
        "document": "[DocumentMarker]",
    },
    "jinaai/jina-colbert-v2-64": {
        "query": "[QueryMarker]",
        "document": "[DocumentMarker]",
    },
}


class HuggingFaceLateEmbedder(LateEmbedder):
    """Late-interaction embedder supporting legacy ColBERT and prefix-based models.

    For models requiring query/document prefixes (e.g. jina-colbert-v2),
    prefixes are resolved automatically from the registry unless overridden.
    Legacy models (colbertv2.0) receive empty prefixes and are unaffected.
    """

    def __init__(
        self,
        model: str,
        batch_size: int = 4,
        device: str = "cpu",
        max_length: Optional[int] = None,
        torch_dtype: Optional[torch.dtype] = None,
        query_prefix: Optional[str] = None,
        document_prefix: Optional[str] = None,
    ) -> None:
        if model not in SUPPORTED_LATE_MODELS:
            raise ValueError(
                f"Model '{model}' is not in the list of supported late-interaction models: "
                f"{sorted(SUPPORTED_LATE_MODELS)}."
            )

        self._model = model
        self._batch_size = batch_size
        self._device = device

        # Resolve prefixes: explicit override > registry > empty string (legacy)
        _registry = _LATE_MODEL_PREFIXES.get(model, {})
        self._query_prefix = (
            query_prefix if query_prefix is not None else _registry.get("query", "")
        )
        self._document_prefix = (
            document_prefix
            if document_prefix is not None
            else _registry.get("document", "")
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._max_length = (
            max_length
            if max_length is not None
            else _resolve_default_max_length(self._tokenizer, fallback=512)
        )

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self._encoder = AutoModel.from_pretrained(model, **model_kwargs).to(device)
        self._encoder.eval()

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        self._dim = cfg.hidden_size

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    # --- Internal helpers -----------------------------------------------

    def _apply_prefix(self, texts: List[str], prefix: str) -> List[str]:
        if not prefix:
            return texts
        return [prefix + t for t in texts]

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
            token_vecs = torch.nn.functional.normalize(token_vecs, p=2, dim=1)
            batch_outputs.append(token_vecs.cpu().tolist())

        return batch_outputs

    def _embed_with_prefix(
        self, texts: List[str], prefix: str, desc: str
    ) -> List[List[List[float]]]:
        prefixed = self._apply_prefix(texts, prefix)
        outputs: List[List[List[float]]] = []
        for i in tqdm(range(0, len(prefixed), self._batch_size), desc=desc):
            outputs.extend(self._encode_batch(prefixed[i : i + self._batch_size]))
        return outputs

    # --- Query-side -----------------------------------------------------

    def embed_query(self, text: str) -> List[List[float]]:
        return self._encode_batch(self._apply_prefix([text], self._query_prefix))[0]

    def embed_queries(self, texts: Iterable[str]) -> List[List[List[float]]]:
        return self._embed_with_prefix(
            list(texts), self._query_prefix, desc="Encoding queries"
        )

    # --- Document-side --------------------------------------------------

    def embed_document(self, text: str) -> List[List[float]]:
        return self._encode_batch(self._apply_prefix([text], self._document_prefix))[0]

    def embed_documents(self, texts: Iterable[str]) -> List[List[List[float]]]:
        return self._embed_with_prefix(
            list(texts), self._document_prefix, desc="Encoding documents"
        )
