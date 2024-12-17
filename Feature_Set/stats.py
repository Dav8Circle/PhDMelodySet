"""
This module contains a series of algorithms that may be used on the different input types
to aid in the calculation of features which use descriptive statistics.
"""
__author__ = "David Whyatt" 

import numpy as np

def range_func(values: list[float]) -> float:
    """Calculates the range (difference between max and min) of a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The range of the values. Returns 0 for empty list.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc
    return float(max(values_array) - min(values_array))

def standard_deviation(values: list[float]) -> float:
    """Calculates the population standard deviation of a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The standard deviation of the values. Returns 0 for empty list.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc
    return float(np.std(values_array, ddof=0))


def shannon_entropy(values: list[float]) -> float:
    """Calculates the Shannon entropy (base-2) of a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The Shannon entropy of the values. Returns 0 for empty list.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Calculate probabilities of each unique value
    _, counts = np.unique(values_array, return_counts=True)
    probabilities = counts / len(values_array)

    # Calculate entropy using the formula: -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def natural_entropy(values: list[float]) -> float:
    """Calculates the natural entropy (base-e) of a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The natural entropy of the values. Returns 0 for empty list.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Calculate probabilities of each unique value
    _, counts = np.unique(values_array, return_counts=True)
    probabilities = counts / len(values_array)

    # Calculate entropy using the formula: -sum(p * ln(p))
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy)


def mean(values: list[float]) -> float:
    """Calculates the arithmetic mean of a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The arithmetic mean. Returns 0 for empty list.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    return float(np.mean(values_array))

def mode(values: list[float]) -> float:
    """Calculates the mode (most frequent value) of a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The mode. If multiple modes exist, returns the smallest one.
        Returns 0 for empty list.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    if len(values_array) == 0:
        return 0.0

    # Get value counts
    unique, counts = np.unique(values_array, return_counts=True)

    # Find max count
    max_count = np.max(counts)

    # Get all modes (values with max count)
    modes = unique[counts == max_count]

    # Return smallest mode
    return float(np.min(modes))

def length(values: list[float]) -> float:
    """Returns the length (number of elements) of a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The length. Returns 0 for empty list.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    return float(len(values_array))
