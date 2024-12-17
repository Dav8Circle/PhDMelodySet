"""
This module contains a series of algorithms that may be called on different input types
to calculate features. These algorithms are rather more specific than the others implemented
in the rest of the repository, and are placed here for clarity and organization,
since they do not group well with the others.
"""
__author__ = "David Whyatt"

import numpy as np

def ratio(x: list[float], y: list[float]) -> list[float]:
    """Calculates the ratio between corresponding elements in two lists.
    
    Args:
        x: First list of numeric values
        y: Second list of numeric values. Must have same length as x.
        
    Returns:
        List containing ratios of corresponding elements (x[i]/y[i]).
        Returns empty list if input lists are empty or have different lengths.
        Returns None for any division by zero.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not x or not y or len(x) != len(y):
        return []

    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Handle division by zero by converting to None
    ratios = []
    for _, (x_val, y_val) in enumerate(zip(x_array, y_array)):
        if y_val == 0 or x_val == 0:
            ratios.append(None)
        else:
            ratios.append(float(x_val / y_val))

    return ratios


def rank_values(values: list[float], descending: bool = False) -> list[float]:
    """Ranks the input values from 1 to n. Ties get the same rank.

    Args:
        values: List of numeric values to rank
        descending: If True, highest value gets rank 1.
                   If False, lowest value gets rank 1.

    Returns:
        List of ranks corresponding to the input values.
        Returns empty list for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return []
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Get sorting order based on descending flag
    order = -1 if descending else 1

    # Sort values and get unique values
    sorted_vals = np.sort(order * values_array)
    unique_vals = np.unique(sorted_vals)

    # Create rank mapping for each unique value
    rank_map = {}
    for i, val in enumerate(unique_vals, 1):
        rank_map[val] = float(i)

    # Map each value to its rank
    ranks = [rank_map[order * val] for val in values_array]

    return ranks

def modulo_twelve(values: list[float]) -> list[float]:
    """Takes modulo 12 of each number in the input list.
    
    Args:
        values: List of numeric values
        
    Returns:
        List with modulo 12 of each value. Returns empty list for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    return [float(x) for x in values_array % 12]

def absolute_values(values: list[float]) -> list[float]:
    """Returns a list with absolute values of all input numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        List containing absolute values of input numbers. Returns empty list for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    return [float(x) for x in np.abs(values_array)]

def limit(values: list[float], x: float = None, y: float = None) -> list[float]:
    """Removes values larger than x or smaller than y from the input list.
    
    Args:
        values: List of numeric values
        x: Upper limit (inclusive). If None, no upper limit is applied.
        y: Lower limit (inclusive). If None, no lower limit is applied.
        
    Returns:
        List containing only values within the specified limits. Returns empty list for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    mask = np.ones(len(values_array), dtype=bool)

    if x is not None:
        mask &= values_array <= x

    if y is not None:
        mask &= values_array >= y

    return [float(x) for x in values_array[mask]]

def nine_percent_significant_values(values: list[float], threshold: float = 0.09) -> list[float]:
    """Returns values that appear more than a given proportion of times in the input list.
    
    Args:
        values: List of numeric values to analyze
        threshold: Minimum proportion (between 0 and 1) required for a value to be 
        considered significant. Default is 0.09 (9%)
                  
    Returns:
        List of values that appear more than the threshold proportion of times.
        Returns empty list for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Get counts of unique values
    unique, counts = np.unique(values_array, return_counts=True)

    # Calculate proportions
    proportions = counts / len(values_array)

    # Get values that exceed threshold
    significant = [float(val) for val, prop in zip(unique, proportions) 
                  if prop >= threshold]

    return significant

def circle_of_fifths(pitch_counts: dict[int, int]) -> dict[int, int]:
    """Reorganizes a dictionary of pitch counts according to the circle of fifths pattern
    (0,7,2,9,4,11,6,1,8,3,10,5).
    
    Args:
        pitch_counts: Dictionary mapping pitch classes (0-12) to their counts
        
    Returns:
        Dictionary with same counts but keys reordered according to circle of fifths.
        Empty input returns empty dictionary.
        
    Raises:
        ValueError: If any pitch class is not a number between 0 and 12
    """
    if not pitch_counts:
        return {}

    # Validate input values
    for pitch in pitch_counts:
        if not isinstance(pitch, int) or pitch < 0 or pitch > 12:
            raise ValueError("Pitch classes must be integers between 0 and 12")

    # Define circle of fifths order
    fifths_order = [0,7,2,9,4,11,6,1,8,3,10,5]

    # Create new dictionary with circle of fifths ordering
    result = {}
    for pitch in fifths_order:
        if pitch in pitch_counts:
            result[pitch] = pitch_counts[pitch]

    return result

def contour_extrema(values: list[float]) -> list[tuple[int, float]]:
    """Finds all contour extremum notes in a sequence.
    
    Identifies:
    - First and last notes
    - Local maxima where surrounding notes are lower
    - Local minima where surrounding notes are higher
    - Handles plateaus by looking at extended context
    
    Args:
        values: List of numeric values
        
    Returns:
        List of tuples (index, value) for each extrema point.
        Returns empty list for empty input or lists shorter than 2 elements.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values or len(values) < 2:
        return []

    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    extrema = [(0, values_array[0])]  # First note is always included
    n = len(values_array)

    # Check each interior point
    for i in range(1, n-1):
        # Get values for comparison
        prev = values_array[i-1]
        curr = values_array[i]
        next_val = values_array[i+1]

        # Simple case: clear maximum or minimum
        if (prev < curr and next_val < curr) or (prev > curr and next_val > curr):
            extrema.append((i, curr))
            continue

        # Handle plateaus by looking at extended context
        if prev == curr or next_val == curr:
            # Look back up to 2 positions if available
            back2 = values_array[i-2] if i >= 2 else None

            # Look forward up to 2 positions if available
            forward2 = values_array[i+2] if i < n-2 else None

            # Check plateau cases
            if prev == curr and forward2 is not None:
                if (back2 is not None and back2 < curr and next_val < curr) or \
                   (back2 is not None and back2 > curr and next_val > curr):
                    extrema.append((i, curr))
            elif next_val == curr and back2 is not None:
                if (forward2 is not None and forward2 < curr and prev < curr) or \
                   (forward2 is not None and forward2 > curr and prev > curr):
                    extrema.append((i, curr))

    extrema.append((n-1, values_array[n-1]))  # Last note is always included
    return [(i, float(v)) for i, v in extrema]  # Ensure values are float

def contour_gradients(values: list[float]) -> list[float]:
    """Calculates gradients between consecutive contour extrema points.
    
    For each pair of consecutive extrema points (ti,pi) and (tj,pj),
    calculates the gradient m = (pj-pi)/(tj-ti)
    
    Args:
        values: List of numeric values
        
    Returns:
        List of gradients between consecutive extrema points.
        Returns empty list for empty input or lists shorter than 2 elements.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values or len(values) < 2:
        return []

    # Get extrema points using existing function
    extrema = contour_extrema(values)

    # Calculate gradients between consecutive extrema
    gradients = []
    for i in range(len(extrema)-1):
        t1, p1 = extrema[i]
        t2, p2 = extrema[i+1]

        # Skip if time difference is zero (shouldn't happen with current extrema logic)
        if t2 - t1 == 0:
            continue

        # Calculate gradient
        gradient = (p2 - p1) / (t2 - t1)
        gradients.append(float(gradient))

    return gradients

def compute_tonality_vector(pitch_classes: list[float]) -> list[float]:
    """Computes the Krumhansl-Schmuckler key-finding correlation vector.
    
    Correlates the distribution of pitch classes in the input with the Krumhansl-Kessler
    key profiles for all 24 possible major and minor keys to determine key likelihood.
    
    Args:
        pitch_classes: List of integers representing pitch classes (0-11)
        
    Returns:
        List of 24 correlation coefficients where:
        - Indices 0-11 correspond to major keys (C, C#, D, etc.)
        - Indices 12-23 correspond to minor keys (c, c#, d, etc.)
        Returns list of zeros if input is empty.
        
    Raises:
        TypeError: If pitch classes cannot be converted to integers between 0-11
    """
    # Krumhansl-Kessler key profiles
    maj_vector = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_vector = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    if not pitch_classes:
        return [0.0] * 24

    # Validate and convert pitch classes to integers
    try:
        pc_ints = [int(pc) for pc in pitch_classes]
        if not all(0 <= pc < 12 for pc in pc_ints):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise TypeError("Pitch classes must be convertible to integers between 0-11") from exc

    # Compute distribution of pitch classes
    pc_dist = [0.0] * 12
    for pc in pc_ints:
        pc_dist[pc] += 1.0

    # Normalize distribution
    total = sum(pc_dist)
    if total > 0:  # Avoid division by zero
        pc_dist = [count / total for count in pc_dist]

    # Initialize correlation vector
    correlations = []

    # Compute correlations for all possible keys
    for i in range(12):  # For each possible root note
        # Rotate profiles to current root
        shifted_maj = maj_vector[-i:] + maj_vector[:-i]
        shifted_min = min_vector[-i:] + min_vector[:-i]

        # Calculate correlations
        maj_corr = np.corrcoef(pc_dist, shifted_maj)
        min_corr = np.corrcoef(pc_dist, shifted_min)

        correlations.append(float(maj_corr))

    # Add minor key correlations
    for i in range(12):
        shifted_min = min_vector[-i:] + min_vector[:-i]
        min_corr = np.corrcoef(pc_dist, shifted_min)
        correlations.append(min_corr if min_corr is not None else 0.0)

    return correlations
