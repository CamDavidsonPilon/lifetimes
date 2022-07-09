.PHONY: init lint check_lint test

init:
	poetry install

lint:
	isort .
	black .

check_lint:
	flake8 .
	isort --check-only .
	black --diff --check --fast .

test:
	pytest