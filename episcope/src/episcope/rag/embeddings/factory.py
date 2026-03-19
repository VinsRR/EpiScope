import logging
from .base import Embedder
from .huggingface import HuggingFaceEmbedder, HuggingFaceSparseEmbedder, HuggingFaceLateEmbedder
from .ollama import OllamaEmbedder
from .openai import OpenAIEmbedder
from .gemini import GeminiEmbedder

logger = logging.getLogger(__name__)

# A set of known OpenAI embedding models to help the factory distinguish them
OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
}

class EmbedderFactory:
    """Instantiates the correct embedder based on the model name."""

    @staticmethod
    def get_embedder(model_name: str, **kwargs) -> Embedder:
        """
        Factory method to get an embedder instance.

        The factory uses heuristics to determine the provider from the model name:
        - Names with '/' are treated as HuggingFace models.
        - Names starting with 'models/' are treated as Gemini models.
        - Names matching a known OpenAI model list are treated as OpenAI models.
        - All other names are assumed to be Ollama models.

        Args:
            model_name: The name of the model.
            **kwargs: Additional arguments to pass to the embedder's constructor.

        Returns:
            An instance of an Embedder subclass.
        """
        if "/" in model_name and not model_name.startswith("models/"):
            logger.info(f"Detected HuggingFace model '{model_name}'. Creating HuggingFaceEmbedder.")
            return HuggingFaceEmbedder(model=model_name, **kwargs)
        elif model_name in OPENAI_EMBEDDING_MODELS:
            logger.info(f"Detected OpenAI model '{model_name}'. Creating OpenAIEmbedder.")
            return OpenAIEmbedder(model=model_name, **kwargs)
        elif "embedding-001" in model_name: # not anymore like this....
            logger.info(f"Detected Gemini model '{model_name}'. Creating GeminiEmbedder.")
            return GeminiEmbedder(model=model_name, **kwargs)
        else:
            logger.info(f"Assuming Ollama model '{model_name}'. Creating OllamaEmbedder.")
            return OllamaEmbedder(model=model_name, **kwargs)

    @staticmethod
    def get_sparse_embedder(model_name: str, **kwargs) -> HuggingFaceSparseEmbedder:
        """Factory method to get a sparse embedder instance."""
        logger.info(f"Creating HuggingFaceSparseEmbedder for model '{model_name}'.")
        return HuggingFaceSparseEmbedder(model=model_name, **kwargs)

    @staticmethod
    def get_late_embedder(model_name: str, **kwargs) -> HuggingFaceLateEmbedder:
        """Factory method to get a late-interaction embedder instance."""
        logger.info(f"Creating HuggingFaceLateEmbedder for model '{model_name}'.")
        return HuggingFaceLateEmbedder(model=model_name, **kwargs)
