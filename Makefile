
.PHONY: all install format lint test run clean help

PY ?= python
PIP ?= pip

install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

format:
	@echo "Formatting with black..."
	black analysis.py test_analysis.py

lint:
	@echo "Linting with flake8..."
	flake8 --max-line-length=100 --ignore=F841,W503 analysis.py test_analysis.py

test:
	@echo "Running tests with coverage..."
	pytest -v -rA -s --maxfail=1 --durations=10 \
		--cov=analysis --cov-report=term-missing --cov-report=html \
		test_analysis.py
	@echo "Coverage HTML report: htmlcov/index.html"

run:
	@echo "Running analysis..."
	$(PY) analysis.py

clean:
	@echo "Cleaning build/test artifacts..."
	rm -rf __pycache__ .pytest_cache htmlcov
	rm -f outputs/*.png outputs/*.txt || true
	rm -f confusion_matrix.png feature_importance.png || true

all: install format lint test run

docker-build:
	docker build -t student-dropout-analysis .

docker-run:
	docker run -it student-dropout-analysis
