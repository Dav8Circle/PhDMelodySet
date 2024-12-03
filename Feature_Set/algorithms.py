import numpy as np
from scipy import stats, signal

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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")

    # Calculate probabilities of each unique value
    unique, counts = np.unique(values_array, return_counts=True)
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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")

    # Calculate probabilities of each unique value
    unique, counts = np.unique(values_array, return_counts=True)
    probabilities = counts / len(values_array)

    # Calculate entropy using the formula: -sum(p * ln(p))
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy)


def distribution_proportions(values: list[float]) -> list[float]:
    """Calculates the proportion of each unique value in a list of numbers.
    
    Args:
        values: List of numeric values
        
    Returns:
        List of proportions for each unique value.
        Returns empty list for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return {}
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")

    # Calculate frequencies of each unique value
    unique, counts = np.unique(values_array, return_counts=True)

    # Calculate proportions
    proportions = counts * (1.0/len(values_array))
    return {float(u): float(p) for u, p in zip(unique, proportions)}


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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Handle division by zero by converting to None
    ratios = []
    for i in range(len(x_array)):
        if y_array[i] == 0 or x_array[i] == 0:
            ratios.append(None)
        else:
            ratios.append(float(x_array[i] / y_array[i]))
            
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
        rank_map[val] = float(i)
        
    # Map each value to its rank
    ranks = [rank_map[order * val] for val in values_array]

    return ranks


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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
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
        Dictionary of consecutive repetition counts for values that appear consecutively more than once.
        Returns empty dictionary for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
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
                repetitions[float(current_value)] = float(current_count)
            current_value = val
            current_count = 1
            
    # Check final run
    if current_count > 1:
        repetitions[float(current_value)] = float(current_count)
            
    return repetitions

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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    return float(len(values_array))

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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    return [float(x) for x in values_array % 12]

def histogram_bins(values: list[float], num_bins: int) -> dict[str, int]:
    """Places data into histogram bins and counts occurrences in each bin.
    
    Creates a histogram by dividing the range of values into equal-width bins
    and counting how many values fall into each bin.
    
    Args:
        values: List of numeric values to bin
        num_bins: Integer specifying number of equal-width bins to create
        
    Returns:
        Dictionary mapping bin range strings (e.g. '1.00-2.00') to counts.
        Returns empty dictionary for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
        ValueError: If num_bins is less than 1
    """
    if not values:
        return {}
        
    if num_bins < 1:
        raise ValueError("Number of bins must be at least 1")
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Calculate histogram
    counts, bin_edges = np.histogram(values_array, bins=num_bins)
    
    # Create dictionary with formatted bin ranges as keys
    result = {}
    for i in range(len(counts)):
        bin_label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
        result[bin_label] = int(counts[i])  # Convert count to integer
    
    return result

def standardize_distribution(values: list[float]) -> list[float]:
    """Converts a list of numbers to a normal distribution with mean 0 and std dev 1.
    
    Applies z-score normalization (standardization) to transform the input values
    into a standard normal distribution.
    
    Args:
        values: List of numeric values to normalize
        
    Returns:
        List of normalized values with mean 0 and standard deviation 1.
        Returns empty list for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
        ValueError: If input has zero standard deviation
    """
    if not values:
        return []
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    mean = np.mean(values_array)
    std = np.std(values_array)
    
    if std == 0:
        raise ValueError("Cannot normalize - standard deviation is zero")
        
    normalized = (values_array - mean) / std
    return [float(x) for x in normalized]

def normalize_distribution(values: list[float]) -> tuple[list[float], float, float, float, float]:
    """Normalizes a list of numbers to a range between 0 and 1 using min-max normalization.
    
    Args:
        values: List of numeric values to normalize
        
    Returns:
        Tuple containing:
        - List of normalized values between 0 and 1
        - Mean of original values
        - Standard deviation of original values
        Returns ([], 0, 0, 0, 0) for empty input.
        
    Raises:
        TypeError: If any element cannot be converted to float
        ValueError: If all input values are identical
    """
    if not values:
        return [], 0.0, 0.0
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    min_val = np.min(values_array)
    max_val = np.max(values_array)
    
    if min_val == max_val:
        raise ValueError("Cannot normalize - all values are identical")
        
    normalized = (values_array - min_val) / (max_val - min_val)
    mean = np.mean(normalized)
    std_dev = np.std(normalized)
    
    return [float(x) for x in normalized], float(mean), float(std_dev)

def kurtosis(values: list[float]) -> float:
    """Calculates the kurtosis (peakedness) of a list of numbers.
    
    Kurtosis measures how heavy-tailed or light-tailed a distribution is
    compared to a normal distribution. Higher kurtosis means more outliers.
    
    Args:
        values: List of numeric values
        
    Returns:
        The kurtosis. Returns 0 for empty list or lists with less than 2 unique values.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Check if there are at least 2 unique values
    if len(np.unique(values_array)) < 2:
        return 0.0
        
    # Use bias=False to get the correct kurtosis for small sample sizes
    return float(stats.kurtosis(values_array, fisher=True, bias=False))

def skew(values: list[float]) -> float:
    """Calculates the skewness (asymmetry) of a list of numbers.
    
    Skewness measures the asymmetry of the probability distribution of a dataset
    around its mean. Positive skew indicates a longer tail on the right,
    negative skew indicates a longer tail on the left.
    
    Args:
        values: List of numeric values
        
    Returns:
        The skewness. Returns 0 for empty list or lists with less than 2 unique values.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values:
        return 0.0
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Check if there are at least 2 unique values
    if len(np.unique(values_array)) < 2:
        return 0.0
        
    # Use bias=False to get the correct skewness for small sample sizes
    return float(stats.skew(values_array, bias=False))

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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    mask = np.ones(len(values_array), dtype=bool)
    
    if x is not None:
        mask &= values_array <= x
        
    if y is not None:
        mask &= values_array >= y
        
    return [float(x) for x in values_array[mask]]

def correlation(x: list[float], y: list[float]) -> float:
    """Calculates the Pearson-Bravais correlation coefficient between two lists of values.
    
    Args:
        x: First list of numeric values
        y: Second list of numeric values. Must have same length as x.
        
    Returns:
        Correlation coefficient between -1 and 1.
        Returns None if input lists are empty or have different lengths.
        Returns 0 if there is no correlation (e.g. if one list has zero variance).
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not x or not y or len(x) != len(y):
        return None
        
    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Check for zero variance in either array
    if np.var(x_array) == 0 or np.var(y_array) == 0:
        return 0.0
        
    # Use numpy's built-in corrcoef function which implements Pearson correlation
    correlation_matrix = np.corrcoef(x_array, y_array)
    
    # corrcoef returns a 2x2 matrix, we want the off-diagonal element
    return float(correlation_matrix[0, 1])

def nine_percent_significant_values(values: list[float], threshold: float = 0.09) -> list[float]:
    """Returns values that appear more than a given proportion of times in the input list.
    
    Args:
        values: List of numeric values to analyze
        threshold: Minimum proportion (between 0 and 1) required for a value to be considered significant.
                  Default is 0.09 (9%)
                  
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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Get counts of unique values
    unique, counts = np.unique(values_array, return_counts=True)
    
    # Calculate proportions
    proportions = counts / len(values_array)
    
    # Get values that exceed threshold
    significant = [float(val) for val, prop in zip(unique, proportions) 
                  if prop >= threshold]
                  
    return significant

def circle_of_fifths(values: list[float]) -> list[float]:
    """Organizes input values into bins according to the circle of fifths pattern (0,6,2,8,4,10,5,11,7,1,9,3).
    
    Args:
        values: List of integers between 0 and 12 to organize
        
    Returns:
        List of counts for each position in circle of fifths order.
        Empty input returns list of zeros.
        
    Raises:
        ValueError: If any value is not an integer between 0 and 12
    """
    if not values:
        return [0.0] * 12
        
    # Validate input values
    for val in values:
        if not isinstance(val, (int, float)) or val < 0 or val > 12:
            raise ValueError("Values must be numbers between 0 and 12")
            
    # Define circle of fifths order
    fifths_order = [0,6,2,8,4,10,5,11,7,1,9,3]
    
    # Initialize result list with zeros
    result = [0.0] * 12
    
    # Count the values according to circle of fifths positions
    for val in values:
        if val < 12:  # Skip 12 since it's equivalent to 0
            result[int(val)] += 1.0
            
    return result

def contour_extrema(values: list[float]) -> list[float]:
    """Finds all contour extremum notes in a sequence.
    
    Identifies:
    - First and last notes
    - Local maxima where surrounding notes are lower
    - Local minima where surrounding notes are higher
    - Handles plateaus by looking at extended context
    
    Args:
        values: List of numeric values
        
    Returns:
        List of extrema values. Returns empty list for empty input
        or lists shorter than 2 elements.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values or len(values) < 2:
        return []
        
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
    
    extrema = [values_array[0]]  # First note is always included
    n = len(values_array)
    
    # Check each interior point
    for i in range(1, n-1):
        # Get values for comparison
        prev = values_array[i-1]
        curr = values_array[i]
        next_val = values_array[i+1]
        
        # Simple case: clear maximum or minimum
        if (prev < curr and next_val < curr) or (prev > curr and next_val > curr):
            extrema.append(curr)
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
                    extrema.append(curr)
            elif next_val == curr and back2 is not None:
                if (forward2 is not None and forward2 < curr and prev < curr) or \
                   (forward2 is not None and forward2 > curr and prev > curr):
                    extrema.append(curr)
    
    extrema.append(values_array[n-1])  # Last note is always included
    return [float(x) for x in extrema]

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
        t1, p1 = i, extrema[i]
        t2, p2 = i+1, extrema[i+1]
        
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
    except (TypeError, ValueError):
        raise TypeError("Pitch classes must be convertible to integers between 0-11")
        
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
        maj_corr = correlation(pc_dist, shifted_maj)  # correlation() now returns float
        min_corr = correlation(pc_dist, shifted_min)
        
        correlations.append(float(maj_corr))
    
    # Add minor key correlations
    for i in range(12):
        shifted_min = min_vector[-i:] + min_vector[:-i]
        min_corr = correlation(pc_dist, shifted_min)
        correlations.append(min_corr if min_corr is not None else 0.0)
    
    return correlations

def yules_k(values: list[float]) -> float:
    """Calculates Yule's K statistic, a measure of lexical diversity.
    Must be called multiple times and averaged over all m-types to match FANTASTIC.
    
    Yule's K is calculated as: K = 10000 * (∑(Vm * m²) - N) / N²
    where:
    - N is the total number of tokens
    - m is the frequency class
    - Vm is the number of types that occur m times
    
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
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
    
    # Count frequency of each unique value
    unique, counts = np.unique(values_array, return_counts=True)
    
    # Create frequency spectrum (how many values appear 1 time, 2 times, etc)
    freq_of_freqs = np.bincount(counts)
    
    # Calculate N (total tokens)
    N = len(values_array)
    
    if N == 0:
        raise ValueError("Cannot calculate Yule's K - no values provided")
    
    # Calculate denominator sum (Vm * m²)
    denom_k = 0
    for m in range(1, len(freq_of_freqs)):
        Vm = freq_of_freqs[m]  # number of types occurring m times
        denom_k += Vm * (m * m)
    
    # Calculate Yule's K with scaling factor of 10000
    k = 10000 * ((denom_k - N) / (N * N))
    
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
    
    # Calculate N (total tokens)
    N = sum(freq_spec.values())
    
    if N == 1:
        return 0.0
        
    # Calculate sum of D components
    D = 0
    for m, Vm in freq_spec.items():
        # Formula: D = sum(Vm * (m/N) * ((m-1)/(N-1)))
        D += Vm * (m/N) * ((m-1)/(N-1))
        
    return float(D)

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
    V = len(unique)
    
    # S = V2/V where V2 is number of types occurring twice
    if V == 0:
        return 0.0
        
    return float(doubles) / V

def honores_h(values: list[float]) -> float:
    """
    Compute Honore's H statistic for a sequence of values.
    Must be called multiple times and averaged over all m-types to match FANTASTIC.

    H = 100 * (log(N) / (1 - (V1/V)))
    where:
    N = total tokens
    V1 = number of types occurring exactly once (hapax legomena)
    V = total number of unique types
    """
    if not values:
        return 0.0
    
    if len(values) == 1:
        return 0.0
        
    # Get frequency spectrum using numpy
    unique, counts = np.unique(values, return_counts=True)
    
    # Calculate components
    V1 = np.sum(counts == 1)  # Number of types occurring once
    V = len(unique)  # Total unique types
    N = len(values)  # Total tokens
    
    # Handle edge cases
    if V1 == 0 or V == 0:
        return 0.0
        
    if V1 == V:
        return 0.0
        
    # Calculate H
    H = 100.0 * (np.log(N) / (1.0 - (float(V1)/V)))
    
    return H
def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Calculate Spearman's rank correlation coefficient between two lists of numbers.
    
    Args:
        x: First list of numeric values
        y: Second list of numeric values
        
    Returns:
        Float value representing Spearman's correlation coefficient.
        Returns 0 if either list is empty or lists have different lengths.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not x or not y or len(x) != len(y):
        return 0.0
        
    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
    
    correlation, _ = stats.spearmanr(x_array, y_array)
    
    # Handle NaN result
    if np.isnan(correlation):
        return 0.0
        
    return correlation

def kendall_tau(x: list[float], y: list[float]) -> float:
    """Calculate Kendall's tau correlation coefficient between two lists of numbers.
    
    Args:
        x: First list of numeric values
        y: Second list of numeric values
        
    Returns:
        Float value representing Kendall's tau correlation coefficient.
        Returns 0 if either list is empty or lists have different lengths.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not x or not y or len(x) != len(y):
        return 0.0
        
    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
    
    tau, _ = stats.kendalltau(x_array, y_array)
    
    # Handle NaN result
    if np.isnan(tau):
        return 0.0
        
    return tau

def diffexp(melody1: list[float], melody2: list[float]) -> float:
    """Calculates the differential expression score between two melodies based on their pitch intervals.
    
    Implements σ(μ₁,μ₂) = e^(-Δp/(N-1)) where Δp is the L1 norm (Manhattan distance) 
    between the pitch interval vectors of the two melodies.
    
    Args:
        melody1: First list of numeric pitch values
        melody2: Second list of numeric pitch values
        
    Returns:
        Float value representing the differential expression score.
        Returns 0.0 if either melody has fewer than 2 notes (no intervals possible).
        
    Raises:
        TypeError: If inputs contain non-numeric values
    """
    # Need at least 2 notes to form intervals
    if len(melody1) < 2 or len(melody2) < 2:
        return 0.0
        
    try:
        # Convert melodies to numpy arrays
        m1 = np.array(melody1, dtype=float)
        m2 = np.array(melody2, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("Melody inputs must contain numeric values")
    
    # Calculate pitch intervals (differences between consecutive notes)
    intervals1 = np.diff(m1)
    intervals2 = np.diff(m2)
    
    # If interval vectors have different lengths, pad shorter one with zeros
    if len(intervals1) != len(intervals2):
        max_len = max(len(intervals1), len(intervals2))
        intervals1 = np.pad(intervals1, (0, max_len - len(intervals1)))
        intervals2 = np.pad(intervals2, (0, max_len - len(intervals2)))
    
    # Calculate Δp as L1 norm (sum of absolute differences) between interval vectors
    delta_p = np.sum(np.abs(intervals1 - intervals2))
    
    # N is the length of the longer melody
    N = max(len(melody1), len(melody2))
    
    # Calculate final score
    score = np.exp(-delta_p / (N - 1))
    
    return float(score)

def diff(melody1: list[float], melody2: list[float]) -> float:
    """Calculates the differential score between two melodies based on their pitch intervals.
    
    Implements σ(μ₁,μ₂) = 1 - Δp/((N-1)Δp∞) where:
    - Δp is the L1 norm (Manhattan distance) between the pitch interval vectors
    - Δp∞ is the maximum absolute interval difference across both melodies
    - N is the length of the longer melody
    
    Args:
        melody1: First list of numeric pitch values
        melody2: Second list of numeric pitch values
        
    Returns:
        Float value representing the differential score.
        Returns 0.0 if either melody has fewer than 2 notes (no intervals possible).
        
    Raises:
        TypeError: If inputs contain non-numeric values
        ValueError: If Δp∞ is zero (no pitch differences between melodies)
    """
    # Need at least 2 notes to form intervals
    if len(melody1) < 2 or len(melody2) < 2:
        return 0.0
        
    try:
        # Convert melodies to numpy arrays
        m1 = np.array(melody1, dtype=float)
        m2 = np.array(melody2, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("Melody inputs must contain numeric values")
    
    # Calculate pitch intervals (differences between consecutive notes)
    intervals1 = np.diff(m1)
    intervals2 = np.diff(m2)
    
    # If interval vectors have different lengths, pad shorter one with zeros
    if len(intervals1) != len(intervals2):
        max_len = max(len(intervals1), len(intervals2))
        intervals1 = np.pad(intervals1, (0, max_len - len(intervals1)))
        intervals2 = np.pad(intervals2, (0, max_len - len(intervals2)))
    
    # Calculate Δp as L1 norm (sum of absolute differences) between interval vectors
    delta_p = np.sum(np.abs(intervals1 - intervals2))
    
    # Calculate Δp∞ as max of absolute intervals across both melodies
    delta_p_inf = max(np.max(np.abs(intervals1)), np.max(np.abs(intervals2)))
    
    if delta_p_inf == 0:
        raise ValueError("Cannot calculate diff score - no pitch differences between melodies")
    
    # N is the length of the longer melody
    N = max(len(melody1), len(melody2))
    
    # Calculate final score
    score = 1 - (delta_p / ((N - 1) * delta_p_inf))
    
    return float(score)

def cross_correlation(x: list[float], y: list[float]) -> list[float]:
    """Calculates the cross-correlation between two lists of numbers using scipy.signal.correlate.
    
    Args:
        x: First list of numeric values
        y: Second list of numeric values
        
    Returns:
        List containing the cross-correlation values. Returns empty list for empty inputs.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not x or not y:
        return []
        
    try:
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Calculate cross-correlation using scipy.signal.correlate
    correlation = signal.correlate(x_array, y_array, mode='full')
    
    return correlation.tolist()
def edit_distance(list1: list[float], list2: list[float], insertion_cost: float=1, deletion_cost: float=1, substitution_cost: float=1) -> float:
    """Calculates the edit distance (Levenshtein distance) between two lists of numbers.
    
    Args:
        list1: First list of numbers
        list2: Second list of numbers
        insertion_cost: Cost of inserting an element (default=1)
        deletion_cost: Cost of deleting an element (default=1) 
        substitution_cost: Cost of substituting an element (default=1)
        
    Returns:
        Float representing the weighted edit distance between the two lists.
    """
    len1, len2 = len(list1), len(list2)
    
    # Create a matrix to store distances
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize the matrix
    for i in range(len1 + 1):
        dp[i][0] = i * deletion_cost
    for j in range(len2 + 1):
        dp[0][j] = j * insertion_cost
    
    # Compute the edit distance
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + deletion_cost,     # Deletion
                              dp[i][j - 1] + insertion_cost,      # Insertion
                              dp[i - 1][j - 1] + substitution_cost) # Substitution
    
    return dp[len1][len2]

def proximity(values: list[float]) -> float:
    """Calculates the proximity score between consecutive notes.
    
    Implements a proximity measure where:
    - Returns 6 for unison (0 semitones)
    - Returns 4 for whole tone (2 semitones)
    - Returns 1 for perfect fourth (5 semitones)
    - Returns 0 for tritone or greater (≥6 semitones)
    
    Args:
        values: List of numeric pitch values
        
    Returns:
        Float representing the proximity score for the last interval.
        Returns 0 for empty list or single value.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if not values or len(values) < 2:
        return 0
        
    try:
        # Get last two values
        last_two = values[-2:]
        interval = abs(last_two[1] - last_two[0])
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
    
    # Calculate proximity score: 6 minus the absolute interval
    proximity_score = 6 - interval
    
    # Return 0 if proximity would be negative (interval ≥ 6)
    return max(0, float(proximity_score))

def registral_return(values: list[float]) -> float:
    """Calculates the registral return score for a sequence of three notes.
    
    Returns:
    - 3 if the pitch returns exactly to the first note
    - 2 if the return is a semitone away from the first note
    - 1 if the return is a tone away from the first note
    - 0 otherwise
    
    The pitch contour must form an arch (up-down or down-up). Returns 0 for
    rising/falling contours or when any pitch is repeated.
    
    Args:
        values: List of numeric pitch values
        
    Returns:
        Float representing the registral return score.
        Returns 0 for lists with fewer than 3 notes.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if len(values) < 3:
        return 0
        
    try:
        # Get the last three notes
        pitch1 = float(values[-3])
        pitch2 = float(values[-2])
        pitch3 = float(values[-1])
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
        
    # Calculate intervals between consecutive notes
    implicative = pitch2 - pitch1  # First interval
    realized = pitch3 - pitch2     # Second interval
    
    # Check if intervals are in different directions (one positive, one negative)
    different_direction = (implicative * realized) < 0
    
    # If either interval is zero (repeated note) or intervals are in same direction
    if implicative == 0 or realized == 0 or not different_direction:
        return 0
        
    # Check if pitch3 is within 2 semitones of pitch1
    if abs(pitch3 - pitch1) <= 2:
        # Calculate return score: 3 - distance from original pitch
        return (3 - abs(pitch3 - pitch1))
    
    return 0

def registral_direction(values: list[float]) -> float:
    """Determines the registral direction based on the last three notes.
    
    Checks if a large interval is followed by a direction change or if a small interval
    is followed by a move in the same direction.
    
    Args:
        values: List of numeric pitch values
        
    Returns:
        Integer 1 if the conditions are met, otherwise 0.
        Returns 0 for lists with fewer than 3 notes.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if len(values) < 3:
        return 0
    
    try:
        # Get the last three notes
        pitch1 = float(values[-3])
        pitch2 = float(values[-2])
        pitch3 = float(values[-1])
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
    
    # Calculate intervals
    implicative = pitch2 - pitch1
    realized = pitch3 - pitch2
    
    # Helper functions
    def large_interval(interval):
        return abs(interval) > 6
    
    def small_interval(interval):
        return abs(interval) < 6
    
    def same_direction(int1, int2):
        return (int1 > 0 and int2 > 0) or (int1 < 0 and int2 < 0)
    
    # Determine registral direction
    if large_interval(implicative):
        return 0 if same_direction(implicative, realized) else 1
    elif small_interval(implicative):
        return 1 if same_direction(implicative, realized) else 0
    else:
        return 0

def intervallic_difference(values: list[float]) -> float:
    """Determines if a large interval is followed by a smaller interval or if a small interval is followed by a similar interval.
    
    Args:
        values: List of numeric pitch values
        
    Returns:
        Integer 1 if the conditions are met, otherwise 0.
        Returns 0 for lists with fewer than 3 notes.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if len(values) < 3:
        return 0
    
    try:
        # Get the last three notes
        pitch1 = float(values[-3])
        pitch2 = float(values[-2])
        pitch3 = float(values[-1])
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
    
    # Calculate intervals
    implicative = pitch2 - pitch1
    realized = pitch3 - pitch2
    
    # Helper functions
    def large_interval(interval):
        return abs(interval) > 6
    
    def small_interval(interval):
        return abs(interval) < 6
    
    def same_direction(int1, int2):
        return (int1 > 0 and int2 > 0) or (int1 < 0 and int2 < 0)
    
    # Determine intervallic difference
    if large_interval(implicative):
        margin = 3 if same_direction(implicative, realized) else 2
        return 1 if abs(realized) < abs(implicative) - margin else 0
    elif small_interval(implicative):
        margin = 3 if same_direction(implicative, realized) else 2
        return 1 if abs(realized) >= abs(implicative) - margin and abs(realized) <= abs(implicative) + margin else 0
    else:
        return 0
    

def closure(values: list[float]) -> float:
    """Calculates the closure score based on the shape defined by the last three notes.
    
    Scores 1 point for a change of direction and 1 point for an interval that is more than a tone smaller than the preceding one.
    
    Args:
        values: List of numeric pitch values
        
    Returns:
        Integer score which can be 0, 1, or 2.
        Returns 0 for lists with fewer than 3 notes.
        
    Raises:
        TypeError: If any element cannot be converted to float
    """
    if len(values) < 3:
        return 0
    
    try:
        # Get the last three notes
        pitch1 = float(values[-3])
        pitch2 = float(values[-2])
        pitch3 = float(values[-1])
    except (TypeError, ValueError):
        raise TypeError("All elements must be numbers")
    
    # Calculate intervals
    implicative = pitch2 - pitch1
    realized = pitch3 - pitch2
    
    # Helper function
    def different_direction(int1, int2):
        return (int1 > 0 and int2 < 0) or (int1 < 0 and int2 > 0)
    
    # Initialize score
    score = 0
    
    # Check conditions
    if different_direction(implicative, realized):
        score += 1
    if abs(realized) < abs(implicative) - 2:
        score += 1
    
    return score

