import logging
from typing import List
import pandas as pd

from .config import OllamaConfig

# Optional dependency
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

logger = logging.getLogger(__name__)

class OllamaCandidateGenerator:
    """Generates reference candidates from table data using an Ollama model."""

    def __init__(self, config: OllamaConfig):
        if not HAS_OLLAMA:
            raise ImportError("ollama is not installed. Please install it with 'pip install ollama'")
        self.config = config
        self.client = ollama.Client()

    def generate(self, df: pd.DataFrame) -> List[str]:
        """
        Takes a DataFrame from an extracted table and returns a list of candidate strings.
        """
        if df.empty:
            return []
            
        try:
            user_content = self.config.user_prompt_template.format(csv_data=df.to_csv(index=False))
            response = self.client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                options={"temperature": self.config.temperature}
            )
            content = response['message']['content']
            if "<no references>" not in content.lower():
                return [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Ollama reference extraction failed: {e}")
        
        return []
