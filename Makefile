PYTHON := python3
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate

# ----------------------
# Install Dependencies
# ----------------------
install:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(ACTIVATE) && pip install --upgrade pip
	@$(ACTIVATE) && pip install -r requirements.txt

# ----------------------
# Convert Notebook → main.py
# ----------------------
export:
	@$(ACTIVATE) && jupyter nbconvert --to script FINAL_CODE.ipynb --output main

# ----------------------
# Run the Notebook as a Script
# ----------------------
run: export
	@$(ACTIVATE) && $(PYTHON) main.py

# ----------------------
# Run the Test Suite
# ----------------------
test: export
	@$(ACTIVATE) && pytest -q

# ----------------------
# Clean Build Files
# ----------------------
clean:
	rm -rf $(VENV)
	rm -rf __pycache__ .pytest_cache *.pyc main.py
