.PHONY: test run clean

run:
	python3 main.py
	python3 src/visualize.py
	python3 src/create_report.py
	@echo "Pipeline complete. Check REPORT.md."

test:
	pytest .

clean:
	rm -f *.png *.csv REPORT.md
