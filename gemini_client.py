import base64
import os
import time
from typing import Dict, Any
import re
from google import genai
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini prompt for skin analysis
PROMPT_TEMPLATE = """You are a specialized medical AI dermatologist. Analyze the provided skin image and determine if it shows any skin conditions or diseases.

Respond *only* in the following format, with no other text:
STATUS: [healthy/unhealthy]
CONFIDENCE: [a float number between 0.0 and 1.0]
DISEASE: [Name of disease, or "None"]
DISEASE_CONFIDENCE: [a float number between 0.0 and 1.0, or 0.0]
"""

# Using Gemini 2.5 Flash model (less overloaded)
MODEL_NAME = "gemini-2.5-flash"

# Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class GeminiClient:
    """
    Handles communication with Google's Gemini 2.5 Flash AI model.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

        if not GOOGLE_API_KEY:
            raise ValueError("Google API Key is missing. Please set GOOGLE_API_KEY in your .env file")

        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        print(f"Gemini client initialized with model: {model_name}")

    def diagnose_skin_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Main method: takes image bytes, returns diagnosis dict
        """
        try:
            # Retry logic for overloaded API
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Convert image bytes to PIL Image for Gemini
                    image = Image.open(io.BytesIO(image_bytes))

                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    print(f"Sending request to Gemini API... (attempt {attempt + 1}/{max_retries})")

                    # Use Gemini's generate_content method
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[
                            PROMPT_TEMPLATE,
                            image
                        ]
                    )

                    # Extract the response text
                    ai_text_response = response.text.strip()
                    print(f"Gemini Response: {ai_text_response}")

                    # Parse the response
                    structured_data = self._parse_response(ai_text_response)

                    # Return
                    return structured_data

                except Exception as e:
                    error_str = str(e).lower()

                    # Check if this is a retryable error
                    if ("503" in error_str or "unavailable" in error_str or "overloaded" in error_str) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                        print(f"Gemini API overloaded, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue

                    # Re-raise the error for our error handler
                    raise e

        except Exception as e:
            error_str = str(e).lower()
            if "503" in error_str or "unavailable" in error_str:
                raise Exception("Gemini API is currently overloaded. The AI model is temporarily unavailable. Please try again in a few minutes.")
            elif "429" in error_str or "quota" in error_str:
                raise Exception("API quota exceeded. Please check your Google AI usage limits.")
            elif "403" in error_str or "forbidden" in error_str:
                raise Exception("API access forbidden. Please check your Google API key permissions.")
            elif "401" in error_str or "unauthorized" in error_str:
                raise Exception("API authentication failed. Please check your GOOGLE_API_KEY in the .env file.")
            else:
                # Generic error for other issues
                raise Exception(f"An error occurred during Gemini analysis: {e}")

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract structured data from AI response using regex.
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
            raise ValueError(f"Failed to parse Gemini response. Raw text: '{response_text}' Error: {e}")
