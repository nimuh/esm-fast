.PHONY: help install install-gpu test test-cpu test-gpu lint format typecheck check clean

help:
	@echo "Targets:"
	@echo "  install      Editable install with dev extras (CPU)"
	@echo "  install-gpu  Editable install with dev + gpu (Triton) extras"
	@echo "  test         Run the full test suite"
	@echo "  test-cpu     Run everything except GPU/Triton tests"
	@echo "  test-gpu     Run only the GPU/Triton tests"
	@echo "  lint         ruff check"
	@echo "  format       ruff format"
	@echo "  typecheck    mypy over the package"
	@echo "  check        lint + typecheck + test-cpu"
	@echo "  clean        Remove build/test caches"

install:
	python -m pip install -e ".[dev]"

install-gpu:
	python -m pip install -e ".[dev,gpu]"

test:
	pytest

test-cpu:
	pytest -m "not gpu"

test-gpu:
	pytest -m gpu

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

check: lint typecheck test-cpu

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
