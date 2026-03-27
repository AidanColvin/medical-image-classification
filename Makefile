.PHONY: test run clean

test:
	@echo "[SYSTEM] Running 25-Test Verification Suite..."
	python3 -m pytest tests/test_comprehensive.py -v | tee test_results.log

run:
	@echo "[SYSTEM] Starting Full Training and Prediction Pipeline..."
	python3 src/train_and_predict.py

clean:
	@echo "[SYSTEM] Cleaning up logs and cache..."
	rm -f test_results.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
