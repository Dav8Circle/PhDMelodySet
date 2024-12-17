"""
This module contains a series of algorithms that may be used on the different input types
to aid in the calculation of features related to complexity.
"""
__author__ = "David Whyatt"

import numpy as np

def yules_k(values: list[float]) -> float:
    """Calculates Yule's K statistic, a measure of lexical diversity.
    Must be called multiple times and averaged over all m-types to match FANTASTIC.
    
    Yule's K is calculated as: K = 10000 * (∑(vm * m²) - n) / n²
    where:
    - n is the total number of tokens
    - m is the frequency class
    - vm is the number of types that occur m times
    
    Args:
        values: List of numeric values
        
    Returns:
        Float value representing Yule's K statistic.
        Returns 0 for empty list or lists with less than 2 elements.
        
    Raises:
        TypeError: If any element cannot be converted to float
        ValueError: If calculation would result in division by zero
    """
    if not values or len(values) < 2:
        return 0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Count frequency of each unique value
    _, counts = np.unique(values_array, return_counts=True)

    # Create frequency spectrum (how many values appear 1 time, 2 times, etc)
    freq_of_freqs = np.bincount(counts)

    # Calculate N (total tokens)
    n = len(values_array)

    if n == 0:
        raise ValueError("Cannot calculate Yule's K - no values provided")

    # Calculate denominator sum (Vm * m²)
    denom_k = 0
    for m in range(1, len(freq_of_freqs)):
        vm = freq_of_freqs[m]  # number of types occurring m times
        denom_k += vm * (m * m)

    # Calculate Yule's K with scaling factor of 10000
    k = 10000 * ((denom_k - n) / (n * n))

    return float(k)

def simpsons_d(values: list[float]) -> float:
    """
    Compute Simpson's D diversity index for a sequence of values.
    Must be called multiple times and averaged over all m-types to match FANTASTIC.
    
    Args:
        values: List of numeric values
        
    Returns:
        float: Simpson's D value
    """
    if not values or len(values) < 2:
        return 0.0

    # Get frequency spectrum using numpy
    unique, counts = np.unique(values, return_counts=True)
    freq_spec = dict(zip(unique, counts))

    # Calculate n (total tokens)
    n = sum(freq_spec.values())

    if n == 1:
        return 0.0

    # Calculate sum of d components
    d = 0
    for m, vm in freq_spec.items():
        # Formula: d = sum(vm * (m/n) * ((m-1)/(n-1)))
        d += vm * (m/n) * ((m-1)/(n-1))

    return float(d)

def sichels_s(values: list[float]) -> float:
    """
    Computes Sichel's S statistic from a list of values.
    S is the proportion of types that occur exactly twice in the sample.
    Must be called multiple times and averaged over all m-types to match FANTASTIC.
    
    Args:
        values: List of values to analyze
        
    Returns:
        float: Sichel's S value
    """
    if not values:
        return 0.0

    if len(values) == 1:
        return 0.0

    # Get frequency spectrum using numpy
    unique, counts = np.unique(values, return_counts=True)

    # Find number of types that occur exactly twice
    doubles = np.sum(counts == 2)

    # Total number of types
    v = len(unique)

    # S = V2/V where V2 is number of types occurring twice
    if v == 0:
        return 0.0

    return float(doubles) / v

def honores_h(values: list[float]) -> float:
    """
    Compute Honore's H statistic for a sequence of values.
    Must be called multiple times and averaged over all m-types to match FANTASTIC.

    H = 100 * (log(n) / (1 - (v1/v)))
    where:
    n = total tokens
    v1 = number of types occurring exactly once (hapax legomena)
    v = total number of unique types
    """
    if not values:
        return 0.0

    if len(values) == 1:
        return 0.0

    # Get frequency spectrum using numpy
    unique, counts = np.unique(values, return_counts=True)

    # Calculate components
    v1 = np.sum(counts == 1)  # Number of types occurring once
    v = len(unique)  # Total unique types
    n = len(values)  # Total tokens

    # Handle edge cases
    if v1 == 0 or v == 0:
        return 0.0

    if v1 == v:
        return 0.0

    # Calculate h
    h = 100.0 * (np.log(n) / (1.0 - (float(v1)/v)))

    return float(h)


def repetition_rate(values: list[float]) -> float:
    """Finds the shortest distance between any repeated value in the list.
    
    Args:
        values: List of numeric values to check for repetitions
        
    Returns:
        Shortest distance between repeats.
        Returns 0 if list is empty or has no repeats.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Track shortest distance found
    min_distance = len(values)
    # Track if we found any repeats
    found_repeat = False

    # Check each value against subsequent values
    for i, _ in enumerate(values_array):
        # Look ahead for first repeat
        for j, val in enumerate(values_array[i + 1:], start=i + 1):
            if values_array[i] == val:
                found_repeat = True
                distance = j - i
                min_distance = min(distance, min_distance)
                # Can break inner loop since we found closest repeat
                break

    return float(min_distance) if found_repeat else 0.0

def repetition_count(values: list[float]) -> list[float]:
    """Counts the number of times each value repeats in the list.
    
    Args:
        values: List of numeric values to count repetitions
        
    Returns:
        List of repetition counts for values that appear more than once.
        Returns empty list for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return {}

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Get counts of each unique value
    unique, counts = np.unique(values_array, return_counts=True)

    # Get counts only for values that repeat (count > 1)
    repeat_counts = [float(count) for count in counts if count > 1]

    return {int(unique[i]): repeat_counts[i] for i in range(len(repeat_counts))}

def consecutive_repetition_count(values: list[float]) -> dict[float, float]:
    """Counts the number of times each value appears consecutively in the list.
    
    Args:
        values: List of numeric values to check for consecutive repetitions
        
    Returns:
        Dictionary of consecutive repetition counts for values that appear consecutively 
        more than once. Returns empty dictionary for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return {}

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    repetitions = {}
    current_value = values_array[0]
    current_count = 1

    # Iterate through values starting from second element
    for val in values_array[1:]:
        if val == current_value:
            current_count += 1
        else:
            # When value changes, record count if > 1 and reset
            if current_count > 1:
                repetitions[float(current_value)] = float(current_count)
            current_value = val
            current_count = 1

    # Check final run
    if current_count > 1:
        repetitions[float(current_value)] = float(current_count)

    return repetitions
