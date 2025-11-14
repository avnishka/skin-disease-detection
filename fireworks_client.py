import base64
import httpx
from PIL import Image
import io
from typing import Dict, Any
import json
import re
import os

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

class FireworksClient:
    """Handles communication with Fireworks AI API"""

    def __init__(self, model_name: str = "accounts/fireworks/models/qwen3-vl-30b-a3b-thinking"):
        """
        Initializes the Fireworks AI client.
        """
        self.model_name = model_name
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.api_key = os.environ.get("FIREWORKS_API_KEY")

        if not self.api_key:
            raise Exception("FIREWORKS_API_KEY environment variable is required")

        self.client = httpx.Client(timeout=120.0)  # Longer timeout for vision models

    def diagnose_skin_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Main method: takes image bytes, returns diagnosis dict
        """
        try:
            # 1. Convert image to URL
            image_url = self._image_to_url(image_bytes)

            # 2. Build the API payload
            payload = self._build_payload(image_url)

            # 3. Send to Fireworks
            raw_ai_text = self._call_fireworks_api(payload)

            # 4. Parse response
            structured_data = self._parse_response(raw_ai_text)

            # 5. Return
            return structured_data

        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error connecting to Fireworks: {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Failed to connect to Fireworks. Check your internet connection. Error: {e}")
        except Exception as e:
            # Catch all other errors (parsing, etc.)
            raise Exception(f"An error occurred during analysis: {e}")

    def _image_to_url(self, image_bytes: bytes) -> str:
        """
        Process image and return data URL for Fireworks API.
        Fireworks may support data URLs directly.
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

        # Convert to base64 for data URL
        base64_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # Return data URL directly - Fireworks might support this
        return f"data:image/jpeg;base64,{base64_data}"


    def _build_payload(self, image_url: str) -> Dict[str, Any]:
        """Create the structured payload for Fireworks API"""
        return {
            "model": self.model_name,
            "max_tokens": 1000,
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.6,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT_TEMPLATE
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ]
        }

    def _call_fireworks_api(self, payload: Dict[str, Any]) -> str:
        """Send request to Fireworks API and get the raw text response"""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = self.client.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()

        raw_data = response.json()
        choices = raw_data.get("choices", [])
        if not choices:
            raise ValueError("Fireworks returned no choices")

        ai_text_response = choices[0].get("message", {}).get("content", "")

        if not ai_text_response:
            raise ValueError("Fireworks returned an empty response.")

        return ai_text_response

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract structured data from AI response using regex.
        """
        try:
            # Remove thinking tags and extract final response
            if "</think>" in response_text:
                response_text = response_text.split("</think>")[-1].strip()

            # Use findall to get all matches, then take the last one (most final)
            status_matches = re.findall(r"STATUS:\s*(healthy|unhealthy)", response_text, re.IGNORECASE)
            # Be more specific - match CONFIDENCE but not DISEASE_CONFIDENCE
            confidence_matches = re.findall(r"(?<!DISEASE_)CONFIDENCE:\s*([\d]+\.[\d]+)", response_text, re.IGNORECASE)
            disease_matches = re.findall(r"DISEASE:\s*([^\n]+?)(?=\s*DISEASE_CONFIDENCE|\s*$)", response_text, re.IGNORECASE)
            disease_conf_matches = re.findall(r"DISEASE_CONFIDENCE:\s*([\d]+\.[\d]+)", response_text, re.IGNORECASE)

            # Take the last match (most final answer)
            status = status_matches[-1].lower() if status_matches else "unhealthy"
            confidence = float(confidence_matches[-1]) if confidence_matches else 0.0
            disease_name = disease_matches[-1].strip() if disease_matches else "Not Specified"
            disease_confidence = float(disease_conf_matches[-1]) if disease_conf_matches else 0.0

            # Set final values
            disease = "None"
            if status == "unhealthy" and disease_name.lower() != "none":
                disease = disease_name

            return {
                "status": status,
                "confidence": confidence,
                "disease": disease,
                "disease_confidence": disease_confidence
            }
        except Exception as e:
            raise ValueError(f"Failed to parse AI response. Raw text: '{response_text}' Error: {e}")
