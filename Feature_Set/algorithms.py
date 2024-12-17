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

def circle_of_fifths(values: list[float]) -> list[float]:
    """Organizes input values into bins according to the circle of fifths pattern 
    (0,6,2,8,4,10,5,11,7,1,9,3).
    
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

def diffexp(melody1: list[float], melody2: list[float]) -> float:
    """Calculates the differential expression score between two melodies
    based on their pitch intervals.
    
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
    except (TypeError, ValueError) as exc:
        raise TypeError("Melody inputs must contain numeric values") from exc

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

    # n is the length of the longer melody
    n = max(len(melody1), len(melody2))

    # Calculate final score
    score = np.exp(-delta_p / (n - 1))

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
    except (TypeError, ValueError) as exc:
        raise TypeError("Melody inputs must contain numeric values") from exc

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

    # n is the length of the longer melody
    n = max(len(melody1), len(melody2))

    # Calculate final score
    score = 1 - (delta_p / ((n - 1) * delta_p_inf))

    return float(score)


def edit_distance(list1: list[float], list2: list[float], insertion_cost: float=1,
                deletion_cost: float=1, substitution_cost: float=1) -> float:
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
