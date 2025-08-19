install:
	uv sync

ui:
	uv run ui.py
	
cli:
	uv run run.py $(ARGS)

typehint:
	uv run mypy src/

lint:
	uv run ruff check src/

format:
	uv run ruff check src/ --fix

clean:
	rm -rf .*_cache coverage.xml .*coverage site report

code-quality: typehint lint clean

.PHONY: checklist