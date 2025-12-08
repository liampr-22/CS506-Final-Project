.PHONY: install run test clean all

# Install all dependencies
install:
	pip install --upgrade pip
	pip install numpy pandas scikit-learn matplotlib seaborn jupyter nbconvert openpyxl pytest

# Run the hyperparameter tuning notebook
run:
	jupyter nbconvert --to notebook --execute hyperparameter_tuning.ipynb --output hyperparameter_tuning_output.ipynb

# Alternative: Run and generate HTML report
run-html:
	jupyter nbconvert --to html --execute hyperparameter_tuning.ipynb

# Run tests
test:
	pytest test_hyperparameter_tuning.py -v

# Clean generated files
clean:
	del /f regression_results.csv clustering_results.csv hyperparameter_tuning_output.ipynb hyperparameter_tuning.html 2>nul

# Do everything: install, test, and run
all: install test run