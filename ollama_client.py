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

    def __init__(
        self, vision_model: str = "qwen3-vl:2b", text_model: str = "llama3.2:3b"
    ):
        """
        Initializes the client with both vision and text models.
        Vision model analyzes images, text model provides additional analysis.
        """
        self.vision_model = vision_model
        self.text_model = text_model
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
        Main method: takes image bytes, returns diagnosis dict using both vision and text models
        """
        try:
            # 1. Convert image
            base64_image = self._image_to_base64(image_bytes)

            # 2. Vision model analysis
            vision_payload = self._build_vision_prompt(base64_image)
            vision_response = self._call_ollama_api(
                vision_payload, model=self.vision_model
            )
            vision_data = self._parse_response(vision_response)

            # 3. Optional text model refinement (if vision model found a disease)
            if (
                vision_data.get("status") == "unhealthy"
                and vision_data.get("disease") != "None"
            ):
                text_payload = self._build_text_refinement_prompt(vision_data)
                text_response = self._call_ollama_api(
                    text_payload, model=self.text_model
                )
                text_data = self._parse_text_refinement(text_response)

                # Combine results - use text model to refine confidence if available
                if text_data.get("refined_confidence"):
                    vision_data["disease_confidence"] = text_data["refined_confidence"]

            # 4. Return combined results
            return vision_data

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

    def _build_vision_prompt(self, base64_image: str) -> Dict[str, Any]:
        """Create the structured payload for vision model analysis"""
        return {
            "model": self.vision_model,
            "messages": [
                {"role": "user", "content": PROMPT_TEMPLATE, "images": [base64_image]}
            ],
            "stream": False,
        }

    def _call_ollama_api(self, payload: Dict[str, Any], model: str = None) -> str:
        """Send request to Ollama chat endpoint and get response"""
        if model:
            payload["model"] = model

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

    def _build_text_refinement_prompt(
        self, vision_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create payload for text model to refine the vision model's diagnosis"""
        disease = vision_data.get("disease", "Unknown")
        confidence = vision_data.get("disease_confidence", 0.0)

        text_prompt = f"""
You are a medical AI assistant. A vision model has diagnosed: {disease} with {confidence*100:.1f}% confidence.

Please refine this diagnosis by:
1. Confirming if this is a reasonable diagnosis for skin conditions
2. Providing a more precise confidence score based on your medical knowledge
3. Suggesting if this could be a different condition

Respond with:
REFINED_CONFIDENCE: [adjusted confidence 0.0-1.0]
REASONING: [brief explanation]
"""

        return {
            "model": self.text_model,
            "messages": [{"role": "user", "content": text_prompt}],
            "stream": False,
        }

    def _parse_text_refinement(self, response_text: str) -> Dict[str, Any]:
        """Parse the text model's refinement response"""
        try:
            confidence_match = re.search(
                r"REFINED_CONFIDENCE: ([\d\.]+)", response_text, re.IGNORECASE
            )
            reasoning_match = re.search(
                r"REASONING: (.+)", response_text, re.IGNORECASE | re.DOTALL
            )

            refined_confidence = (
                float(confidence_match.group(1)) if confidence_match else None
            )
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            return {"refined_confidence": refined_confidence, "reasoning": reasoning}
        except Exception as e:
            # If parsing fails, return empty dict - vision model result will be used as-is
            return {}
