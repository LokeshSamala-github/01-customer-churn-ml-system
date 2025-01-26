.PHONY: install data train explain serve test lint fmt all clean

install:
	pip install -r requirements-dev.txt

data:
	python data/generate_data.py --rows 7000

train:
	PYTHONPATH=src python -m churn_system.train --search-iters 20

explain:
	PYTHONPATH=src python -m churn_system.explain --sample-size 800

serve:
	PYTHONPATH=src uvicorn churn_system.api:app --reload --port 8000

test:
	pytest -v

lint:
	ruff check .

fmt:
	ruff check . --fix

# Full pipeline from scratch, in order.
all: data train explain test

clean:
	rm -rf data/warehouse.duckdb .pytest_cache .ruff_cache **/__pycache__
