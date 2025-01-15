"""
This module contains a series of algorithms that may be used on the different input types
to aid in the calculation of features related to complexity.
"""
__author__ = "David Whyatt"

import numpy as np

def yules_k(values: list[float]) -> float:
    """Calculates Yule's K statistic, a measure of lexical diversity.

    Must be called multiple times and averaged over all m-types to match FANTASTIC.
    
    Yule's K is calculated as: K = 1000 * (∑(vm * m²) - n) / n²
    where:
    - n is the total number of tokens
    - m is the frequency class
    - vm is the number of types that occur m times

    Therefore, K tends to be higher where there is less lexical diversity.

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze

    Returns
    -------
    float
        Value representing Yule's K statistic.
        Returns 0 for empty list or lists with less than 2 elements.

    Raises
    ------
    TypeError
        If any element cannot be converted to float
    ValueError
        If calculation would result in division by zero

    Examples
    --------
    >>> yules_k([1, 2, 2, 3, 3, 3])  # Values with different frequencies
    37.037...
    >>> yules_k([1, 1, 1])  # All same value
    222.22...
    >>> yules_k([])  # Empty list
    0
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

    # Calculate Yule's K with scaling factor of 1000 and normalization
    k = (1.0 / abs(n)) * 1000 * ((denom_k - n) / (n * n))

    return float(k)

def simpsons_d(values: list[float]) -> float:
    """Compute Simpson's D diversity index for a sequence of values.

    Must be called multiple times and averaged over all m-types to match FANTASTIC.

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze

    Returns
    -------
    float
        Simpson's D value. Returns 0.0 for empty list or lists with less than 2 elements.

    Examples
    --------
    >>> simpsons_d([1, 2, 2, 3, 3, 3])  # Values with different frequencies
    0.733...
    >>> simpsons_d([1, 1, 1])  # All same value
    0.0
    >>> simpsons_d([])  # Empty list
    0.0
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
    """Computes Sichel's S statistic from a list of values.

    S is the proportion of types that occur exactly twice in the sample.
    Must be called multiple times and averaged over all m-types to match FANTASTIC.

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze

    Returns
    -------
    float
        Sichel's S value. Returns 0.0 for empty list or single-element list.

    Examples
    --------
    >>> sichels_s([1, 2, 2, 3, 3, 3])  # One type occurs twice
    0.333...
    >>> sichels_s([1, 1, 2, 2, 3, 3])  # All types occur twice
    1.0
    >>> sichels_s([])  # Empty list
    0.0
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
    """Compute Honore's H statistic for a sequence of values.

    Must be called multiple times and averaged over all m-types to match FANTASTIC.
    H = 100 * (log(n) / (1 - (v1/v)))
    where:
    - n = total tokens
    - v1 = number of types occurring exactly once (hapax legomena)
    - v = total number of unique types

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze

    Returns
    -------
    float
        Honore's H value. Returns 0.0 for empty list or single-element list.

    Examples
    --------
    >>> honores_h([1, 2, 2, 3, 3, 3])  # One hapax legomenon
    268.763...
    >>> honores_h([1, 1, 1])  # No hapax legomena
    0.0
    >>> honores_h([])  # Empty list
    0.0
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
    """Calculates the average distance between all repeated values in the list.

    Parameters
    ----------
    values : list[float]
        List of numeric values to check for repetitions

    Returns
    -------
    float
        Average distance between repeated values.
        Returns 0 if list is empty or has no repeats.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> repetition_rate([1, 2, 1, 3, 2])  # Multiple repeats with different distances
    2.5
    >>> repetition_rate([1, 2, 3])  # No repeats
    0.0
    >>> repetition_rate([])  # Empty list
    0.0
    """
    if not values:
        return 0.0

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Track all distances between repeats
    distances = []

    # Check each value against subsequent values
    for i, _ in enumerate(values_array):
        # Look ahead for all repeats
        for j, val in enumerate(values_array[i + 1:], start=i + 1):
            if values_array[i] == val:
                distance = j - i
                distances.append(distance)

    # Calculate average distance if we found any repeats
    return float(np.mean(distances)) if distances else 0.0

def repetition_count(values: list[float]) -> list[float]:
    """Counts the number of times each value repeats in the list.

    Parameters
    ----------
    values : list[float]
        List of numeric values to count repetitions

    Returns
    -------
    dict[int, float]
        Dictionary mapping values to their repetition counts for values that appear more than once.
        Returns empty dictionary for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> repetition_count([1, 2, 2, 3, 3, 3])  # Multiple repeats
    {2: 2.0, 3: 3.0}
    >>> repetition_count([1, 2, 3])  # No repeats
    {}
    >>> repetition_count([])  # Empty list
    {}
    """
    if not values:
        return {}

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Get counts of each unique value
    unique, counts = np.unique(values_array, return_counts=True)

    # Create list of indices where counts > 1
    repeat_indices = [i for i, count in enumerate(counts) if count > 1]
    
    # Return dictionary mapping values to their counts
    return {int(unique[i]): float(counts[i]) for i in repeat_indices}

def consecutive_repetition_count(values: list[float]) -> dict[float, float]:
    """Counts the number of times each value appears consecutively in the list.

    Parameters
    ----------
    values : list[float]
        List of numeric values to check for consecutive repetitions

    Returns
    -------
    dict[float, float]
        Dictionary mapping values to their consecutive repetition counts for values that appear
        consecutively more than once. Returns empty dictionary for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> consecutive_repetition_count([1, 1, 2, 2, 2, 3])  # Multiple consecutive repeats
    {1.0: 2.0, 2.0: 3.0}
    >>> consecutive_repetition_count([1, 2, 1, 2])  # No consecutive repeats
    {}
    >>> consecutive_repetition_count([])  # Empty list
    {}
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

def consecutive_fifths(values: list[float]) -> dict[float, int]:
    """Checks the input list for consecutive values separated by perfect fifths.

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze

    Returns
    -------
    dict[float, int]
        Dictionary mapping starting values to their consecutive fifths counts.
        Returns empty dictionary for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> consecutive_fifths([1, 8, 15, 22])  # Consecutive fifths
    {1.0: 3}
    >>> consecutive_fifths([1, 2, 3, 4])  # No consecutive fifths
    {}
    >>> consecutive_fifths([])  # Empty list
    {}
    """
    # return sum([(j - i) % 12 == 7 for i, j in zip(values, values[1:])])
    if not values:
        return {}

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    fifths = {}
    current_value = values_array[0]
    current_count = 0

    # Iterate through values starting from second element
    for val in values_array[1:]:
        if (val - current_value) % 12 == 7:
            current_count += 1
            current_value = val
        else:
            # When value changes, record count if > 0 and reset
            if current_count > 0:
                fifths[float(values_array[0])] = current_count
            current_value = val
            current_count = 0

    # Check final run
    if current_count > 0:
        fifths[float(values_array[0])] = current_count

    return fifths