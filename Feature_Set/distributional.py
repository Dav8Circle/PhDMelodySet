"""
This module contains a series of algorithms that may be used on the different input types
to aid in the calculation of features related to distributional properties.
"""
__author__ = "David Whyatt"

import numpy as np
from scipy import stats

def histogram_bins(values: list[float], num_bins: int) -> dict[str, int]:
    """Places data into histogram bins and counts occurrences in each bin.

    Parameters
    ----------
    values : list[float]
        List of numeric values to bin
    num_bins : int
        Number of equal-width bins to create

    Returns
    -------
    dict[str, int]
        Dictionary mapping bin range strings (e.g. '1.00-2.00') to counts.
        Returns empty dictionary for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float
    ValueError
        If num_bins is less than 1

    Examples
    --------
    >>> histogram_bins([1, 2, 3, 4, 5], 2)
    {'1.00-3.00': 2, '3.00-5.00': 3}
    >>> histogram_bins([], 5)  # Empty list
    {}
    >>> histogram_bins([1, 1, 2, 2, 3], 3)
    {'1.00-1.67': 2, '1.67-2.33': 2, '2.33-3.00': 1}
    """
    if not values:
        return {}

    if num_bins < 1:
        raise ValueError("Number of bins must be at least 1")

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Calculate histogram
    counts, bin_edges = np.histogram(values_array, bins=num_bins)

    # Create dictionary with formatted bin ranges as keys
    result = {}
    for i, (count, edge) in enumerate(zip(counts, bin_edges)):
        bin_label = f"{edge:.2f}-{bin_edges[i+1]:.2f}"
        result[bin_label] = int(count)  # Convert count to integer

    return result

def standardize_distribution(values: list[float]) -> list[float]:
    """Converts a list of numbers to a normal distribution with mean 0 and std dev 1.

    Parameters
    ----------
    values : list[float]
        List of numeric values to normalize

    Returns
    -------
    list[float]
        List of normalized values with mean 0 and standard deviation 1.
        Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float
    ValueError
        If input has zero standard deviation

    Examples
    --------
    >>> standardize_distribution([1, 2, 3])
    [-1.224..., 0.0, 1.224...]
    >>> standardize_distribution([]) # Empty list
    []
    >>> standardize_distribution([10, 20, 30])
    [-1.224..., 0.0, 1.224...]
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    mean_val = np.mean(values_array)
    std = np.std(values_array)

    if std == 0:
        raise ValueError("Cannot normalize - standard deviation is zero")

    normalized = (values_array - mean_val) / std
    return [float(x) for x in normalized]

def normalize_distribution(values: list[float]) -> tuple[list[float], float, float]:
    """Normalizes a list of numbers to a range between 0 and 1 using min-max normalization.

    Parameters
    ----------
    values : list[float]
        List of numeric values to normalize

    Returns
    -------
    tuple[list[float], float, float]
        Tuple containing:
        - List of normalized values between 0 and 1
        - Mean of normalized values
        - Standard deviation of normalized values
        Returns ([], 0.0, 0.0) for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float
    ValueError
        If all input values are identical

    Examples
    --------
    >>> normalize_distribution([1, 2, 3])  # doctest: +ELLIPSIS
    ([0.0, 0.5, 1.0], 0.5, 0.408...)
    >>> normalize_distribution([])  # Empty list
    ([], 0.0, 0.0)
    >>> normalize_distribution([10, 20, 30])  # doctest: +ELLIPSIS
    ([0.0, 0.5, 1.0], 0.5, 0.408...)
    """
    if not values:
        return [], 0.0, 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    min_val = np.min(values_array)
    max_val = np.max(values_array)

    if min_val == max_val:
        raise ValueError("Cannot normalize - all values are identical")

    normalized = (values_array - min_val) / (max_val - min_val)
    mean_val = np.mean(normalized)
    std_dev = np.std(normalized)

    return [float(x) for x in normalized], float(mean_val), float(std_dev)

def kurtosis(values: list[float]) -> float:
    """Calculates the kurtosis (peakedness) of a list of numbers.

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    float
        The kurtosis. Returns 0 for empty list or lists with less than 2 unique values.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> kurtosis([1, 2, 2, 3])
    1.5
    >>> kurtosis([]) # Empty list
    0.0
    >>> kurtosis([1, 1, 1]) # Less than 2 unique values
    0.0
    """
    if not values:
        return 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Check if there are at least 2 unique values
    if len(np.unique(values_array)) < 2:
        return 0.0

    # Use bias=False to get the correct kurtosis for small sample sizes
    return float(stats.kurtosis(values_array, fisher=True, bias=False))

def skew(values: list[float]) -> float:
    """Calculates the skewness (asymmetry) of a list of numbers.

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    float
        The skewness. Returns 0 for empty list or lists with less than 2 unique values.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> skew([1, 2, 2, 3])  # doctest: +ELLIPSIS
    0.0
    >>> skew([])  # Empty list
    0.0
    >>> skew([1, 1, 1])  # Less than 2 unique values
    0.0
    """
    if not values:
        return 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Check if there are at least 2 unique values
    if len(np.unique(values_array)) < 2:
        return 0.0

    # Use bias=False to get the correct skewness for small sample sizes
    return float(stats.skew(values_array, bias=False))

def distribution_proportions(values: list[float]) -> dict[float, float]:
    """Calculates the proportion of each unique value in a list of numbers.

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    dict[float, float]
        Dictionary mapping unique values to their proportions.
        Returns empty dict for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> distribution_proportions([1, 2, 2, 3])
    {1.0: 0.25, 2.0: 0.5, 3.0: 0.25}
    >>> distribution_proportions([])  # Empty list
    {}
    >>> distribution_proportions([1.5, 1.5, 2.5])
    {1.5: 0.6666666666666666, 2.5: 0.3333333333333333}
    """
    if not values:
        return {}
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Calculate frequencies of each unique value
    unique, counts = np.unique(values_array, return_counts=True)

    # Calculate proportions
    proportions = counts * (1.0/len(values_array))
    return {float(u): float(p) for u, p in zip(unique, proportions)}
