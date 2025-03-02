.PHONY: FORCE install lint format test clean

install: FORCE
	pip install -e ".[dev]"

lint: FORCE
	ruff check .
	black --check .
	mypy .

format: FORCE
	ruff check --fix .
	black .

test: FORCE
	pytest -v test/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name "*.eggs" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 
	
FORCE: