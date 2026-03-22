from typing import Any, Dict, List, Optional, Tuple
from common.ai_model.ai_interface import AIModelInterface
from common.utils import parse_detections
import os
from google import genai
from google.genai import types
import json
import re
from dotenv import load_dotenv

load_dotenv()


class GeminiModel(AIModelInterface):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=api_key)
        self.model = model_name

    def generate_from_image(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)

    def generate_json_content(self, prompt: str) -> Dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2, response_mime_type="application/json"
            ),
        )

        return json.loads(response.text)


geminiModel = GeminiModel()
