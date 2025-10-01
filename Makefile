.PHONY: venv install clean test index run

# Create virtual environment
venv:
	python3 -m venv crisis-env
	@echo "✅ Virtual environment created. Activate with: source crisis-env/bin/activate"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Build RAG index from manuals
index:
	python scripts/build_index.py --input data/manuals --output data/index
	@echo "✅ RAG index built"

# Run text mode
run:
	python main.py

# Run voice mode
voice:
	python main.py --voice

# Run tests
test:
	pytest tests/ -v

# Clean generated files
clean:
	rm -rf data/index/*
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	@echo "✅ Cleaned generated files"

# Download models (optional - you already have them!)
models:
	@echo "Models already downloaded. Skipping..."

# Format code
format:
	black src/ scripts/ tests/ main.py
