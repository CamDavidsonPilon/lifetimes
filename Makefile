autopep8:
	autopep8 --ignore E501,E241,W690 --in-place --recursive --aggressive lifetimes/

# F401=imported but unused, E501=line too long
lint:
	flake8 --ignore F401,E501 lifetimes

autolint: autopep8 lint
