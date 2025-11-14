# Skin Disease Detection

AI-powered skin disease detection web application using **Fireworks AI** and FastAPI.

**Default AI Backend: Fireworks AI** - No local model installation required, just add your API key!

## Setup

### Quick Setup (Recommended)
Use the provided Makefile for easy setup:
```bash
# Full setup (dependencies + environment setup)
make setup

# Or install components separately:
make install-deps  # Install Python dependencies
make setup-env     # Setup environment variables (.env file)
```

### Manual Setup
1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up Fireworks AI API:
   ```bash
   # Copy the example environment file
   cp env_example.txt .env

   # Edit .env and add your Fireworks API key from https://fireworks.ai/
   # Replace 'your_fireworks_api_key_here' with your actual API key
   ```

   The app uses **Fireworks AI** with advanced vision models:
   - **Vision Model** (`qwen3-vl-30b-a3b-thinking`): Analyzes skin images using advanced computer vision
   - **Note:** Images are processed locally and sent as data URLs to Fireworks API

3. Run the application:
   ```bash
   uv run python app.py
   ```

4. Open http://127.0.0.1:8000 in your browser

### Makefile Commands
- `make setup` - Complete setup (dependencies + environment variables)
- `make install-deps` - Install Python dependencies only
- `make setup-env` - Setup environment variables (.env file)
- `make run` - Start the FastAPI application
- `make clean` - Remove cache files
- `make help` - Show all available commands

## Features

- Upload skin images for AI analysis
- **Powered by Fireworks AI** with state-of-the-art vision models
- Uses Qwen3-VL-30B-A3B-Thinking model for accurate skin condition analysis
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
