PYTHON := python3
VENV := .venv
ACTIVATE := . $(VENV)/bin/activate

install:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(ACTIVATE) && pip install --upgrade pip
	@$(ACTIVATE) && pip install -r requirements.txt

export:
	@$(ACTIVATE) && jupyter nbconvert --to script FINAL_CODE.ipynb --output main

run: export
	@$(ACTIVATE) && $(PYTHON) main.py

test: export
	@$(ACTIVATE) && pytest -q

clean:
	rm -rf $(VENV)
	rm -rf __pycache__ .pytest_cache *.pyc main.py
