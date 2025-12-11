PYTHON := python3
VENV := .venv

install:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(VENV)/bin/pip install --upgrade pip
	@$(VENV)/bin/pip install -r requirements.txt

export:
	jupyter nbconvert --to script FINAL_CODE.ipynb --output main

run: export
	$(PYTHON) main.py

test: export
	pytest -q

clean:
	rm -rf $(VENV)
	rm -rf __pycache__ .pytest_cache *.pyc main.py
