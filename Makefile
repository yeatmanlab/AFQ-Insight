flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 flake8 --ignore N802,N806,W503 `find . -name \*.py | grep -v setup.py | grep -v /doc/ | grep -v /.ipynb_checkpoints/`; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

test:
    # Unit testing using pytest
	py.test --pyargs afqinsight --cov-report term-missing --cov-config .coveragerc --cov=afqinsight

devtest:
    # Unit testing with the -x option, aborts testing after first failure
    # Useful for development when tests are long
	py.test -x --pyargs afqinsight --cov-report term-missing --cov-config .coveragerc --cov=afqinsight
