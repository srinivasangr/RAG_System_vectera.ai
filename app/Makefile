.PHONY: install ingest run eval test lint clean snowflake-init

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

snowflake-init:
	python -m rag_system.storage.init_snowflake

ingest:
	python -m rag_system.ingest.pipeline

run:
	streamlit run rag_system/ui/streamlit_app.py

eval:
	python -m eval.run_eval

test:
	pytest -v

lint:
	ruff check rag_system/ eval/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
