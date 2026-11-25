PYTHON ?= python3

.PHONY: install test compile

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

compile:
	$(PYTHON) -m compileall -q rascal_lab scripts rascal_loop midnight mom95_mlp45 mom95_mlp45_kernel whale_donor control_NOCHANGE
