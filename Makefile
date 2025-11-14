# Skin Disease Detection - Makefile (Fireworks AI)

.PHONY: help install-deps setup-env setup run clean

# Default target
help:
	@echo "Skin Disease Detection with Fireworks AI"
	@echo "Available commands:"
	@echo "  install-deps    Install Python dependencies using uv"
	@echo "  setup-env       Setup Fireworks AI API key (.env file)"
	@echo "  setup          Full setup (dependencies + Fireworks API)"
	@echo "  run            Start the FastAPI application"
	@echo "  clean          Remove cache files"
	@echo "  help           Show this help message"

# Install Python dependencies
install-deps:
	uv sync

# Setup environment variables
setup-env:
	@echo "Setting up environment variables..."
	@if [ ! -f .env ]; then \
		cp env_example.txt .env; \
		echo "Created .env file. Please edit it and add your FIREWORKS_API_KEY from https://fireworks.ai/"; \
	else \
		echo ".env file already exists."; \
	fi

# Full setup (dependencies + environment)
setup: install-deps setup-env

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
