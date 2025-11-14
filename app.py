# src/skin_checker/app.py

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from models import DiagnosisResponse, ErrorResponse
from fireworks_client import FireworksClient
import logging
from typing import Union
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI Backend Configuration ---
# Fireworks AI is used for vision analysis
AI_BACKEND = "FIREWORKS"

# File validation settings
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"]
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- App Initialization ---
app = FastAPI(
    title="Skin Disease Checker API",
    description="AI-powered skin image analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_ai_client():
    """Factory function to get the appropriate AI client."""
    if AI_BACKEND == "FIREWORKS":
        return FireworksClient()
    else:
        raise ValueError(
            f"Invalid AI_BACKEND: {AI_BACKEND}. Only FIREWORKS is supported."
        )

# Initialize a single, reusable instance of the AI client
try:
    ai_client = get_ai_client()
    logger.info(f"AI client initialized successfully (Backend: {AI_BACKEND}).")
except Exception as e:
    logger.error(f"Failed to initialize AI client: {e}")
    ai_client = None

# --- API Endpoints ---

@app.get("/", include_in_schema=False)
async def root():
    """Serves the main HTML page."""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "skin-disease-checker"}


@app.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request: Invalid file"},
        413: {"model": ErrorResponse, "description": "Payload Too Large: File exceeds size limit"},
        422: {"model": ErrorResponse, "description": "Unprocessable Entity: Invalid file type"},
        500: {"model": ErrorResponse, "description": "Internal Server Error: AI analysis failed"},
        503: {"model": ErrorResponse, "description": "Service Unavailable: AI client not ready"}
    }
)
async def diagnose_skin(
    file: UploadFile = File(...)
) -> DiagnosisResponse:
    """
    Main endpoint: accepts a skin image, validates it, and returns an
    AI-powered diagnosis.
    """
    
    if not ai_client:
        logger.error("Attempted to use /diagnose, but AI client failed to initialize.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured. Please contact the administrator."
        )
        
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided."
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid file type. Please upload a JPEG, PNG, or WEBP image."
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        logger.warning(f"File uploaded exceeds size limit: {len(image_bytes)} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_PAYLOAD_TOO_LARGE,
            detail=f"File size exceeds limit of {MAX_FILE_SIZE_MB} MB."
        )
    
    logger.info(f"Received file: {file.filename}, Size: {len(image_bytes)} bytes")

    try:
        # Call the synchronous method from our client
        diagnosis_data = ai_client.diagnose_skin_image(image_bytes)
        
        # Validate the data with our Pydantic model
        response = DiagnosisResponse(**diagnosis_data)
        logger.info(f"Successfully diagnosed image: {file.filename}")
        
        return response

    except Exception as e:
        logger.error(f"AI analysis failed for file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during AI analysis. Error: {e}"
        )

if __name__ == "__main__":
    # This part runs if you were to execute `python app.py`
    # Run the FastAPI app directly
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
