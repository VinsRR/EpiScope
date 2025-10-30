import logging
from typing import List
import pandas as pd

from .config import OllamaConfig, GeminiConfig, GeminiCredentials

# Optional dependency
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

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

class GeminiFullFileGenerator:
    """Generates reference candidates from a full PDF file using Gemini."""

    def __init__(self, config: GeminiConfig, creds: GeminiCredentials):
        if not HAS_GEMINI:
            raise ImportError("google-generativeai is not installed. Please install it with 'pip install google-generativeai'")
        self.config = config
        genai.configure(api_key=creds.api_key)
        self.model = genai.GenerativeModel(self.config.model, system_instruction=self.config.system_prompt)

    def generate(self, pdf_path: str) -> List[str]:
        try:
            logger.info(f"Uploading {pdf_path} to Gemini...")
            uploaded_file = genai.upload_file(path=pdf_path, display_name=pdf_path)
            
            response = self.model.generate_content(
                [self.config.user_prompt, uploaded_file],
                generation_config={"temperature": self.config.temperature}
            )
            
            content = response.text
            return [line.strip() for line in content.splitlines() if line.strip()]

        except Exception as e:
            logger.error(f"Gemini reference extraction failed: {e}")
        
        return []
