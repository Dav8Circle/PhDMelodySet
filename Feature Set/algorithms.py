import numpy as np


def range_func(values):
    if not values:
        return 0
    if not all(isinstance(x, int) for x in values):
        raise TypeError("All elements must be integers")
    return max(values) - min(values)


def standard_deviation(values):
    if not values:
        return 0
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
    return np.std(values_array, ddof=0)


def shannon_entropy(values):
    if not values:
        return 0
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")

    # Calculate probabilities of each unique value
    unique, counts = np.unique(values_array, return_counts=True)
    probabilities = counts / len(values_array)

    # Calculate entropy using the formula: -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def natural_entropy(values):
    if not values:
        return 0
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")

    # Calculate probabilities of each unique value
    unique, counts = np.unique(values_array, return_counts=True)
    probabilities = counts / len(values_array)

    # Calculate entropy using the formula: -sum(p * ln(p))
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy


def distribution_proportions(values):
    if not values:
        return {}
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")

    # Calculate frequencies of each unique value
    unique, counts = np.unique(values_array, return_counts=True)

    # Create dictionary mapping values to their proportions
    # (frequencies scaled from 0 to 1)
    dist = {float(val): count * (1.0/len(values_array))
            for val, count in zip(unique, counts)}
    return dist


def rank_values(values, descending=False):
    """Ranks the input values from 1 to n. Ties get the same rank.

    Args:
        values: List of numeric values to rank
        descending: If True, highest value gets rank 1.
                   If False, lowest value gets rank 1.

    Returns:
        List of ranks corresponding to the input values
    """
    if not values:
        return []
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")

    # Get sorting order based on descending flag
    order = -1 if descending else 1
    
    # Sort values and get unique values
    sorted_vals = np.sort(order * values_array)
    unique_vals = np.unique(sorted_vals)
    
    # Create rank mapping for each unique value
    rank_map = {}
    for i, val in enumerate(unique_vals, 1):
        rank_map[val] = i
        
    # Map each value to its rank
    ranks = [rank_map[order * val] for val in values_array]

    return ranks


def repetition_rate(values):
    """Finds the shortest distance between any repeated value in the list.
    
    Args:
        values: List of numeric values to check for repetitions
        
    Returns:
        Integer representing shortest distance between repeats.
        Returns 0 if list is empty or has no repeats.
    """
    if not values:
        return 0
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Track shortest distance found
    min_distance = len(values)
    # Track if we found any repeats
    found_repeat = False
    
    # Check each value against subsequent values
    for i in range(len(values_array)):
        # Look ahead for first repeat
        for j in range(i + 1, len(values_array)):
            if values_array[i] == values_array[j]:
                found_repeat = True
                distance = j - i
                min_distance = min(distance, min_distance)
                # Can break inner loop since we found closest repeat
                break
                
    return min_distance if found_repeat else 0


def repetition_count(values):
    """Counts the number of times each value repeats in the list.
    
    Args:
        values: List of numeric values to count repetitions
        
    Returns:
        Dictionary mapping values to their repetition counts.
        Only includes values that appear more than once.
    """
    if not values:
        return {}
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Get counts of each unique value
    unique, counts = np.unique(values_array, return_counts=True)
    
    # Create dict of only values that repeat (count > 1)
    repetitions = {}
    for val, count in zip(unique, counts):
        if count > 1:
            repetitions[float(val)] = int(count)
            
    return repetitions

def consecutive_repetition_count(values):
    """Counts the number of times each value appears consecutively in the list.
    
    Args:
        values: List of numeric values to check for consecutive repetitions
        
    Returns:
        Dictionary mapping values to their consecutive repetition counts.
        Only includes values that appear consecutively more than once.
    """
    if not values:
        return {}
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
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
                repetitions[float(current_value)] = int(current_count)
            current_value = val
            current_count = 1
            
    # Check final run
    if current_count > 1:
        repetitions[float(current_value)] = int(current_count)
            
    return repetitions

def mean(values):
    """Calculates the arithmetic mean of a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The arithmetic mean of the values. Returns 0 for empty list.
    """
    if not values:
        return 0
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    return float(np.mean(values_array))

def mode(values):
    """Calculates the mode (most frequent value) of a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The mode of the values. If multiple modes exist, returns the smallest one.
        Returns 0 for empty list.
    """
    if not values:
        return 0
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    if len(values_array) == 0:
        return 0
        
    # Get value counts
    unique, counts = np.unique(values_array, return_counts=True)
    
    # Find max count
    max_count = np.max(counts)
    
    # Get all modes (values with max count)
    modes = unique[counts == max_count]
    
    # Return smallest mode
    return float(np.min(modes))

