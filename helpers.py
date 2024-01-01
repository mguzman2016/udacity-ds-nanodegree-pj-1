import numpy as np

def percentage_to_float(column):
    """
    Convert a column of percentage strings to floating-point values.

    This function converts strings representing percentages (e.g., '50%') in a pandas Series
    to their corresponding floating-point values (e.g., 0.5). It handles missing values by
    replacing them with 0.

    Parameters:
    column (pandas.Series): A Series containing percentage strings.

    Returns:
    pandas.Series: A Series with percentages as floats.
    """
    replaced_column = column.str.rstrip('%').astype('float') / 100
    return replaced_column.fillna(0)

def price_to_float(column):
    """
    Convert a column of price strings to floating-point values.

    This function transforms strings representing prices (e.g., '$1,000') in a pandas Series
    to float values (e.g., 1000.0). It removes currency symbols and commas, and fills missing
    values with 0.

    Parameters:
    column (pandas.Series): A Series containing price strings.

    Returns:
    pandas.Series: A Series with prices as floats.
    """
    replaced_column = column.str.replace('[$,]', '', regex=True).astype('float')
    return replaced_column.fillna(0)

def process_dict_column(column):
    """
    Convert dictionary-like strings in a column to lists.

    This function processes a pandas Series containing dictionary-like string representations
    (e.g., '{"key": "value"}') and converts each string into a list of key-value pairs.

    Parameters:
    column (pandas.Series): A Series with dictionary-like strings.

    Returns:
    pandas.Series: A Series where each dictionary-like string is converted to a list.
    """
    return column.str.replace('[{}"]', '', regex=True).str.split(',')

def replace_with_median(column):
    """
    Replace missing values in a column with the column's median.

    This function fills missing values (NaNs) in a pandas Series with the median value of
    the column.

    Parameters:
    column (pandas.Series): A Series to process.

    Returns:
    pandas.Series: The Series with NaNs replaced by the median value.
    """
    median = column.median()
    return fill_with_value(column,median)

def replace_with_mode(column):
    """
    Replace missing values in a column with the column's mode.

    This function fills missing values (NaNs) in a pandas Series with the most frequently
    occurring value (mode) of the column.

    Parameters:
    column (pandas.Series): A Series to process.

    Returns:
    pandas.Series: The Series with NaNs replaced by the mode value.
    """
    mode = column.mode()[0]
    return fill_with_value(column,mode)

def fill_with_value(column,value):
    """
    Fill missing values in a column with a specified value.

    This function replaces missing values (NaNs) in a pandas Series with a given value.

    Parameters:
    column (pandas.Series): A Series to process.
    value (scalar): The value used to replace NaNs.

    Returns:
    pandas.Series: The Series with NaNs replaced by the specified value.
    """
    return column.fillna(value)

def apply_log_transform(column):
    """
    Apply a logarithmic transformation to a pandas Series.

    This function applies a natural logarithm transformation (log(1 + x)) to each element
    of the input Series. It is useful for normalizing data and reducing skewness.

    Parameters:
    column (pandas.Series): A Series to transform.

    Returns:
    pandas.Series: The Series with the logarithmic transformation applied.
    """
    return np.log1p(column)
