import base64
import httpx
from PIL import Image
import io
from typing import Dict, Any
import json
import re

# This is the prompt that forces the AI to give us the
# "classification" output you wanted.
PROMPT_TEMPLATE = """
You are a specialized medical AI. Analyze the provided image of a skin condition.
Respond *only* in the following format, with no other text:
STATUS: [healthy/unhealthy]
CONFIDENCE: [a float number between 0.0 and 1.0]
DISEASE: [Name of disease, or "None"]
DISEASE_CONFIDENCE: [a float number between 0.0 and 1.0, or 0.0]
"""

class OllamaClient:
    """Handles communication with a local Ollama model"""

    def __init__(self, model_name: str = "qwen3-vl:2b"):
        """
        Initializes the client.
        Note: Using qwen3-vl:2b for image analysis (supports vision).
        """
        self.model_name = model_name
        self.base_url = "http://127.0.0.1:11434"
        self.api_url = f"{self.base_url}/api/chat"
        self.client = httpx.Client(timeout=60.0)
        self._check_connection()

    def _check_connection(self):
        """Checks if the Ollama server is running."""
        try:
            response = self.client.get(self.base_url)
            response.raise_for_status()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            raise Exception(f"Failed to connect to Ollama at {self.base_url}. Is it running? Error: {e}")

    def diagnose_skin_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Main method: takes image bytes, returns diagnosis dict
        """
        try:
            # 1. Convert image
            base64_image = self._image_to_base64(image_bytes)

            # 2. Build the API payload
            payload = self._build_prompt(base64_image)

            # 3. Send to Ollama
            raw_ai_text = self._call_ollama_api(payload)

            # 4. Parse response
            structured_data = self._parse_response(raw_ai_text)

            # 5. Return
            return structured_data

        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error connecting to Ollama: {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Failed to connect to Ollama. Is it running? Error: {e}")
        except Exception as e:
            # Catch all other errors (parsing, etc.)
            raise Exception(f"An error occurred during analysis: {e}")

    def _image_to_base64(self, image_bytes: bytes) -> str:
        """
        Convert image bytes to base64 string, resizing/compressing if needed.
        """
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        max_size_kb = 500
        quality = 85
        while True:
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG", quality=quality)
            size_kb = len(img_buffer.getvalue()) / 1024

            if size_kb <= max_size_kb or quality <= 10:
                break

            width, height = img.size
            img = img.resize((int(width * 0.9), int(height * 0.9)), Image.LANCZOS)
            quality -= 5

        return base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    def _build_prompt(self, base64_image: str) -> Dict[str, Any]:
        """Create the structured payload for Ollama /api/chat endpoint"""
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE,
                    "images": [base64_image]
                }
            ],
            "stream": False
        }

    def _call_ollama_api(self, payload: Dict[str, Any]) -> str:
        """Send request to Ollama chat endpoint and get response"""
        response = self.client.post(self.api_url, json=payload)
        response.raise_for_status()

        raw_data = response.json()
        message = raw_data.get("message", {})
        ai_text_response = message.get("content", "")

        if not ai_text_response:
            raise ValueError("AI returned an empty response.")

        return ai_text_response

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract structured data from AI response using regex.
        """
        try:
            status_match = re.search(r"STATUS: (healthy|unhealthy)", response_text, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE: ([\d\.]+)", response_text, re.IGNORECASE)
            disease_match = re.search(r"DISEASE: ([^\n]+?)(?=\n|$)", response_text, re.IGNORECASE)
            disease_conf_match = re.search(r"DISEASE_CONFIDENCE: ([\d\.]+)", response_text, re.IGNORECASE)

            status = status_match.group(1).lower() if status_match else "unhealthy"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0

            disease = "None"
            disease_confidence = 0.0

            if status == "unhealthy":
                disease_name = disease_match.group(1).strip() if disease_match else "Not Specified"
                if disease_name.lower() != "none":
                    disease = disease_name
                    disease_confidence = float(disease_conf_match.group(1)) if disease_conf_match else 0.0

            return {
                "status": status,
                "confidence": confidence,
                "disease": disease,
                "disease_confidence": disease_confidence
            }
        except Exception as e:
            raise ValueError(f"Failed to parse AI response. Raw text: '{response_text}' Error: {e}")
