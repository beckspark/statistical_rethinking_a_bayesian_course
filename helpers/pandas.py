# Helper functions for pandas/DataFrame data
import pandas as pd

# Helper function to "standardize" variables
def standardize(series: pd.Series):
    """Standardize a pandas series"""
    std_series = (series - series.mean()) / series.std()
    return std_series
