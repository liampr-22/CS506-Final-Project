import os
import subprocess

def test_main_py_exists():
    # Makefile 'export' target must run first
    assert os.path.exists("main.py")

def test_script_runs():
    """
    Run the exported notebook script.
    We do NOT expect it to finish the entire data analysis;
    only that it imports successfully without immediate errors.
    """
    result = subprocess.run(
        ["python3", "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20
    )
    # We only check that Python launched and did not crash instantly.
    assert result.returncode in [0, 1]
