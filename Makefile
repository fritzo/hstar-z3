.PHONY: all
all: lint

.PHONY: install
install: FORCE
	pip install -e ".[dev]"

.PHONY: lint
lint: FORCE
	ruff check .
	black --check .
	mypy --install-types --non-interactive .
	nbqa mypy .

.PHONY: format
format: FORCE
	black .
	ruff check --fix .
	nbstripout notebooks/*.ipynb

.PHONY: test
test: lint FORCE
	pytest -v test/

.PHONY: clean
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
	
.PHONY: FORCE
FORCE:
