VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(PYTHON) -m pip

.PHONY: all setup install_nvtop
.SILENT: all setup install_nvtop

all:
	echo "|> No target selected. Abort."

setup:
	echo "=========| setup started... |========="
	echo "|> cleaning old virtual environment..."
	rm -rf $(VENV)
	echo "|> creating virtual environment..."
	python3 -m venv $(VENV)
	echo "|> upgrading pip..."
	$(PIP) install --upgrade pip
	echo "|> installing requirements..."
	$(PIP) install -r requirements.txt
	echo "|> creating the .venv kernel for jupyter notebook..."
	$(PYTHON) -m ipykernel install --user --name=$(VENV) --display-name=$(VENV)
	echo "=========| setup completed |========="
