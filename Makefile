install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 *.py --ignore=E501,W503

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format lint