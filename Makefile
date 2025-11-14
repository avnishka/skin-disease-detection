# Skin Disease Detection - Makefile

.PHONY: help install-deps install-model setup run clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install-deps    Install Python dependencies using uv"
	@echo "  install-model   Pull the Qwen3-VL-2B model from Ollama"
	@echo "  setup          Full setup (dependencies + model)"
	@echo "  run            Start the FastAPI application"
	@echo "  clean          Remove cache files"
	@echo "  help           Show this help message"

# Install Python dependencies
install-deps:
	uv sync

# Install the Qwen vision model
install-model:
	@echo "Installing Qwen3-VL-2B model..."
	ollama pull qwen3-vl:2b
	@echo "Model installed successfully!"

# Full setup (dependencies + model)
setup: install-deps install-model

# Run the application
run:
	uv run python app.py

# Clean cache files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
