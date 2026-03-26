export PYTHONPATH := $(shell pwd)

.PHONY: run submit clean

run:
	python3 main.py

submit:
	python3 src/generate_submission.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf data/submissions/*
