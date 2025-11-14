# Skin Disease Detection

AI-powered skin disease detection web application using FastAPI and Ollama vision models.

## Setup

### Quick Setup (Recommended)
Use the provided Makefile for easy setup:
```bash
# Full setup (dependencies + Ollama model)
make setup

# Or install components separately:
make install-deps  # Install Python dependencies
make install-model # Pull the Qwen3-VL-2B model
```

### Manual Setup
1. Install dependencies:
   ```bash
   uv sync
   ```

2. Install and set up Ollama:
   ```bash
   # Install Ollama from: https://ollama.ai/

   # Pull the Qwen3-VL-2B vision model specifically for image analysis:
   ollama pull qwen3-vl:2b

   # Verify the model is available:
   ollama list
   ```

   The app is configured to use Ollama with the `qwen3-vl:2b` vision model for image analysis.

3. Run the application:
   ```bash
   make run
   # Or manually: uv run python app.py
   ```

4. Open http://127.0.0.1:8000 in your browser

### Makefile Commands
- `make setup` - Complete setup (dependencies + model)
- `make install-deps` - Install Python dependencies only
- `make install-model` - Pull Qwen3-VL-2B model only
- `make run` - Start the FastAPI application
- `make clean` - Remove cache files
- `make help` - Show all available commands

## Features

- Upload skin images for AI analysis
- Uses Qwen3-VL-2B vision model for accurate image understanding
- Returns structured diagnosis with confidence scores
- FastAPI backend with modern web interface
- Supports JPEG, PNG, and WebP images up to 10MB

## API Response Format

The AI returns diagnoses in this format:
```
STATUS: [healthy/unhealthy]
CONFIDENCE: [0.0-1.0]
DISEASE: [Disease name or "None"]
DISEASE_CONFIDENCE: [0.0-1.0]
```
