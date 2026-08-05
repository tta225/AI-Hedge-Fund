.PHONY: install dev test lint fmt type check demo terminal clean

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev,yfinance,agents]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

fmt:
	python -m ruff format src tests
	python -m ruff check --fix src tests

type:
	python -m mypy

check: lint type test

demo:
	python -m axiom.cli demo

terminal:
	python -m axiom.cli terminal

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
