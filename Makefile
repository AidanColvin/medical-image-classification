export PYTHONPATH := $(shell pwd)

.PHONY: test run submit clean

test:
	pytest tests/ -W ignore::RuntimeWarning

run:
	python3 main.py

submit:
	python3 src/generate_submission.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
