def test_imports():
    import pandas
    import numpy
    import matplotlib
    import seaborn
    import plotly
    import scipy
    import sklearn
    import xgboost

def test_notebook_exists():
    import os
    assert os.path.exists("FINAL_CODE.ipynb")
