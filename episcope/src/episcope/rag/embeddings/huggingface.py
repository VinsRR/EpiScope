from __future__ import annotations
from typing import Iterable, List, Dict, Any, Optional, Sequence, Tuple
from tqdm import tqdm
import torch

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

from .base import Embedder

SUPPORTED_DENSE_MODELS = {
    # Small / robust defaults
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    # Retrieval-oriented BGE family
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    # E5 family
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
}

SUPPORTED_SPARSE_MODELS = {
    "naver/splade-cocondenser-ensembledistil",
    "naver/splade-cocondenser-selfdistil",
    "naver/splade-v3-distilbert",
}

SUPPORTED_LATE_MODELS = {
    "colbert-ir/colbertv2.0",
}

SUPPORTED_CROSS_ENCODER_MODELS = {
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L4-v2",
}

def _resolve_default_max_length(tokenizer, fallback: Optional[int] = None) -> Optional[int]:
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is None:
        return fallback

    # HF uses huge sentinels when no meaningful limit is set
    if isinstance(model_max_length, int) and model_max_length > 100_000:
        return fallback

    return model_max_length



class HuggingFaceEmbedder(Embedder):
    """Dense text embedder backed by SentenceTransformer.

    This wrapper is intentionally limited to known-good single-vector embedding
    models that fit the library's dense retrieval blueprint.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 8,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        normalize: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        trust_remote_code: bool = True,
        prompts: Optional[dict[str, str]] = None,
        default_prompt_name: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if model not in SUPPORTED_DENSE_MODELS:
            raise ValueError(
                f"Model '{model}' is not in the list of supported dense embedding models: "
                f"{sorted(SUPPORTED_DENSE_MODELS)}."
            )

        self._model = model
        self._batch_size = batch_size
        self._device = device
        self._normalize = normalize
        self._precision = precision
        self._truncate_dim = truncate_dim

        self._encoder = SentenceTransformer(
            model_name_or_path=model,
            device=device,
            trust_remote_code=trust_remote_code,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs or {},
            tokenizer_kwargs=tokenizer_kwargs or {},
            config_kwargs=config_kwargs or {},
        )

        # Respect an explicit user choice; otherwise inherit the model default.
        if max_length is not None:
            self._encoder.max_seq_length = max_length

        self._max_length = self._encoder.get_max_seq_length()

        # Infer embedding dimension from one actual forward pass, which is robust
        # even when configs or truncation settings differ from expectations.
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


# class HuggingFaceEmbedder(Embedder):
#     """Wrapper around a HuggingFace embedding model."""

#     def __init__(
#         self,
#         model: str = "sentence-transformers/all-MiniLM-L6-v2",
#         batch_size: int = 8,
#         device: Optional[str] = None,
#         max_length: Optional[int] = None,
#         normalize: bool = True,
#         trust_remote_code: bool = True,
#         show_progress_bar: bool = False,
#         **model_kwargs: Any,
#     ) -> None:
#         if model not in SUPPORTED_DENSE_MODELS:
#             raise ValueError(
#                 f"Model '{model}' is not in the list of supported dense embedding models: "
#                 f"{sorted(SUPPORTED_DENSE_MODELS)}."
#             )
#         self._model = model
#         self._batch_size = batch_size
#         self._device = device
#         self._max_length = max_length
#         self._normalize = normalize

#         self._embedder = HuggingFaceEmbedding(
#             model_name=model,
#             device=device,
#             max_length=max_length,
#             normalize=normalize,
#             embed_batch_size=batch_size,
#             trust_remote_code=trust_remote_code,
#             show_progress_bar=show_progress_bar,
#             **model_kwargs,
#         )
#         self._dim = len(self.embed_text("test"))

#     @property
#     def model_name(self) -> str:
#         return self._model

#     @property
#     def dim(self) -> int:
#         return self._dim

#     def embed_text(self, text: str) -> List[float]:
#         return self._embedder.get_text_embedding(text)

#     def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
#         texts_list = list(texts)
#         vectors: List[List[float]] = []
#         for i in tqdm(range(0, len(texts_list), self._batch_size), desc="Embedding texts"):
#             batch = texts_list[i : i + self._batch_size]
#             vectors.extend(self._embedder.get_text_embedding_batch(batch))
#         return vectors



class HuggingFaceSparseEmbedder:
    """Generic sparse embedder wrapper (e.g. SPLADE-like models)."""

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
        self._max_length = max_length if max_length is not None else _resolve_default_max_length(self._tokenizer)

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self._encoder = AutoModelForMaskedLM.from_pretrained(model, **model_kwargs).to(device)
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
        for i in tqdm(range(0, len(texts_list), self._batch_size), desc="Embedding sparse texts"):
            batch = texts_list[i : i + self._batch_size]
            outputs.extend(self._encode_batch(batch))
        return outputs


class HuggingFaceLateEmbedder:
    """Late-interaction embedder for early ColBERT-compatible models only."""

    def __init__(
        self,
        model: str,
        batch_size: int = 4,
        device: str = "cpu",
        max_length: Optional[int] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        assert model in SUPPORTED_LATE_MODELS, (
            f"HuggingFaceLateEmbedder currently supports only early ColBERT-style models: "
            f"{sorted(SUPPORTED_LATE_MODELS)}. Got: {model}"
        )

        self._model = model
        self._batch_size = batch_size
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._max_length = max_length if max_length is not None else _resolve_default_max_length(
            self._tokenizer, fallback=512
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

    def embed_text(self, text: str) -> List[List[float]]:
        return self._encode_batch([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[List[float]]]:
        texts_list = list(texts)
        outputs: List[List[List[float]]] = []
        for i in tqdm(range(0, len(texts_list), self._batch_size), desc="Embedding late-interaction texts"):
            batch = texts_list[i:i + self._batch_size]
            outputs.extend(self._encode_batch(batch))
        return outputs


class HuggingFaceCrossEncoderReranker:
    """Cross-encoder reranker over retrieved candidates."""

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: Optional[int] = None,
        trust_remote_code: bool = False,
        # backend: str = "torch",
        # model_kwargs: Optional[Dict[str, Any]] = None,
        # tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model = model
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

        self._reranker = CrossEncoder(
            model_name=model,
            device=device,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            # backend=backend,
            # model_kwargs=model_kwargs,
            # tokenizer_kwargs=tokenizer_kwargs,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        scores = self._reranker.predict(
            list(pairs),
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        *,
        text_key: str = "text",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, cand[text_key]) for cand in candidates if text_key in cand and isinstance(cand[text_key], str)]
        usable_candidates = [cand for cand in candidates if text_key in cand and isinstance(cand[text_key], str)]

        if not usable_candidates:
            raise ValueError(
                f"Cross-encoder reranking requires a '{text_key}' field with string content in candidates."
            )

        scores = self.score_pairs(pairs)

        reranked: List[Dict[str, Any]] = []
        for cand, score in zip(usable_candidates, scores):
            item = dict(cand)
            item["cross_score"] = score
            # Do NOT overwrite "score" so that similarity_score is preserved for thresholding
            reranked.append(item)

        reranked.sort(key=lambda x: x.get("cross_score", float('-inf')), reverse=True)
        return reranked[:top_k] if top_k is not None else reranked