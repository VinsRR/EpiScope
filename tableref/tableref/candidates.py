import logging
import json
import re
import time
from typing import List, Optional
import pandas as pd

from .config import OllamaConfig, GeminiConfig, GeminiCredentials, IncludedStudiesPayload

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

try:
    from json_repair import loads as json_repair_loads
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

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
        if not HAS_JSON_REPAIR:
            raise ImportError("json_repair is not installed. Please install it with 'pip install json_repair'")
        self.config = config
        genai.configure(api_key=creds.api_key)
        self.model = genai.GenerativeModel(self.config.model)

    def generate(self, pdf_path: str) -> Optional[IncludedStudiesPayload]:
        max_retries = 1
        for attempt in range(max_retries):
            try:
                logger.info(f"Uploading {pdf_path} to Gemini...")
                uploaded_file = genai.upload_file(path=pdf_path, display_name=pdf_path)

                self.model.system_instruction = self.config.system_prompt

                response = self.model.generate_content(
                    [self.config.user_prompt, uploaded_file],
                    generation_config={"temperature": self.config.temperature},
                    request_options={"timeout": 600}
                )

                raw_text = response.text
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    json_start = raw_text.find('{')
                    json_end = raw_text.rfind('}')
                    if json_start != -1 and json_end != -1:
                        json_text = raw_text[json_start:json_end+1]
                    else:
                        logger.error(f"No JSON object found in response from Gemini for {pdf_path}")
                        raise ValueError("No JSON object found in response")

                payload_dict = json_repair_loads(json_text)

                # Handle cases where studies are dicts like [{'citation': '...'}]
                if payload_dict.get("studies") and isinstance(payload_dict["studies"], list):
                    if all(isinstance(s, dict) and "citation" in s for s in payload_dict["studies"]):
                        logger.info("Found 'studies' as a list of citation dicts; extracting strings.")
                        payload_dict["studies"] = [s["citation"] for s in payload_dict["studies"]]

                # Handle cases where notes is a string instead of a list
                if payload_dict.get("notes") and isinstance(payload_dict["notes"], str):
                    logger.info("Found 'notes' as a string; wrapping in a list.")
                    payload_dict["notes"] = [payload_dict["notes"]]

                # Handle case where declared_included_count is an int instead of a dict
                count_val = payload_dict.get("declared_included_count")
                if isinstance(count_val, int):
                    logger.info("Found 'declared_included_count' as an int; wrapping in a dict.")
                    payload_dict["declared_included_count"] = {"value": count_val}

                return IncludedStudiesPayload.model_validate(payload_dict)

            except Exception as e:
                logger.error(f"Gemini reference extraction failed for {pdf_path} on attempt {attempt + 1}: {e}")
                if 'response' in locals() and hasattr(response, 'text'):
                    logger.error(f"Raw Gemini response on failure: {response.text}")

                if attempt < max_retries - 1:
                    sleep_time = 10 * (attempt + 1)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {max_retries} retries failed for {pdf_path}.")

        return None

    def __str__(self):
        return "GeminiFullFileGenerator"
