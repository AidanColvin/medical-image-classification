test:
	pytest test_suite.py -W ignore::RuntimeWarning

run: test
	python3 main.py

ensemble:
	python3 ensemble_submissions.py

clean:
	rm -rf __pycache__ .pytest_cache
	rm -f submission_fold_*.csv
