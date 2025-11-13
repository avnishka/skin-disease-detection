import base64
import httpx
from PIL import Image
import io
from typing import Dict, Any
import json
import re

# We use the same prompt, but as a "question" for the VQA model
PROMPT_TEMPLATE = """
You are a specialized medical AI. Analyze the provided image of a skin condition.
Respond *only* in the following format, with no other text:
STATUS: [healthy/unhealthy]
CONFIDENCE: [a float number between 0.0 and 1.0]
DISEASE: [Name of disease, or "None"]
DISEASE_CONFIDENCE: [a float number between 0.0 and 1.0, or 0.0]
"""

# --- !!! IMPORTANT !!! ---
# 1. Paste your Hugging Face "Read" token here.
# 2. Get this from https://huggingface.co/settings/tokens
HF_API_TOKEN = ".."
# ---

# We will use a Hugging Face VQA (Visual Question Answering) model
HF_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


class HuggingFaceClient:
    """
    Handles communication with the AI model.
    NOTE: This class has been MODIFIED to use the
    Hugging Face Inference API instead of Ollama.
    """

    def __init__(self, model_name: str = HF_MODEL_ID):
        self.model_name = model_name
        # This is the Hugging Face Inference API URL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.client = httpx.Client(timeout=60.0)
        
        if not HF_API_TOKEN or HF_API_TOKEN == "YOUR_HUGGING_FACE_API_TOKEN_GOES_HERE":
            raise ValueError("Hugging Face API Token is missing in huggingface_client.py")
            
        self.headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    def diagnose_skin_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Main method: takes image bytes, returns diagnosis dict
        """
        try:
            # 1. Convert image to base64
            base64_image = self._image_to_base64(image_bytes)
            
            # 2. Build the API payload for Hugging Face
            payload = self._build_prompt(base64_image)
            
            # 3. Send to HF API
            raw_ai_text = self._call_hf_api(payload)
            
            # 4. Parse response (this method is unchanged)
            structured_data = self._parse_response(raw_ai_text)
            
            # 5. Return
            return structured_data
            
        except httpx.HTTPStatusError as e:
            # Handle API errors
            error_data = e.response.json()
            error_msg = error_data.get("error", str(e))
            if "currently loading" in str(error_msg):
                error_msg = "AI model is loading. Please try again in 20 seconds."
            raise Exception(f"Hugging Face API error: {error_msg}")
        except httpx.RequestError as e:
            raise Exception(f"Failed to connect to Hugging Face API. Check your internet. Error: {e}")
        except Exception as e:
            # Catch all other errors (parsing, etc.)
            raise Exception(f"An error occurred during analysis: {e}")

    def _image_to_base64(self, image_bytes: bytes) -> str:
        """
        Convert image bytes to base64 string.
        (This method is unchanged, HF API also needs base64)
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
        """
        Create the structured *payload* for the Hugging Face VQA API.
        """
        return {
            "inputs": {
                # We send our prompt as the "question"
                "question": PROMPT_TEMPLATE,
                # And the image as base64
                "image": base64_image
            }
        }

    def _call_hf_api(self, payload: Dict[str, Any]) -> str:
        """
        Send request to Hugging Face API and get the raw text response
        (Method name is kept for compatibility with app.py)
        """
        # Tell the API to wait for the model to load if it's not ready
        params = {"wait_for_model": True}
        
        response = self.client.post(
            self.api_url, 
            headers=self.headers, 
            json=payload,
            params=params
        )
        response.raise_for_status()  # Raises error for 4xx/5xx
        
        raw_data = response.json()
        
        # HF VQA returns a list, e.g., [{"generated_text": "..."}]
        if not raw_data or not isinstance(raw_data, list) or "generated_text" not in raw_data[0]:
            raise ValueError(f"Unexpected API response from Hugging Face: {raw_data}")
            
        ai_text_response = raw_data[0].get("generated_text", "")
        
        # The 'llava' model often includes the prompt in its response.
        # We must clean it.
        if PROMPT_TEMPLATE in ai_text_response:
             # Get only the text *after* the prompt
            ai_text_response = ai_text_response.split(PROMPT_TEMPLATE)[-1].strip()
        
        if not ai_text_response:
            raise ValueError("AI returned an empty response.")
            
        return ai_text_response

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract structured data from AI response using regex.
        (This method is 100% UNCHANGED, as it's parsing the
         text we asked the AI to generate.)
        """
        try:
            # Use regex to find the key-value pairs
            status_match = re.search(r"STATUS: (healthy|unhealthy)", response_text, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE: ([\d\.]+)", response_text, re.IGNORECASE)
            disease_match = re.search(r"DISEASE: ([\w\s]+)", response_text, re.IGNORECASE)
            disease_conf_match = re.search(r"DISEASE_CONFIDENCE: ([\d\.]+)", response_text, re.IGNORECASE)

            # Extract values, providing defaults if not found
            status = status_match.group(1).lower() if status_match else "unhealthy"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0
            
            disease = "None"
            disease_confidence = 0.0

            # Only look for disease if status is unhealthy
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
            # If parsing fails, we raise an error
            raise ValueError(f"Failed to parse AI response. Raw text: '{response_text}' Error: {e}")