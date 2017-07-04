autopep8:
	autopep8 --ignore E501,E241,W690 --in-place --recursive --aggressive lifetimes/

lint:
	flake8 lifetimes

autolint: autopep8 lint

pycodestyle:
	pycodestyle lifetimes

pydocstyle:
	pydocstyle lifetimes