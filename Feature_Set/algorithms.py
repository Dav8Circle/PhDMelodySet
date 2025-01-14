"""
This module contains a series of algorithms that may be called on different input types
to calculate features. These algorithms are rather more specific than the others implemented
in the rest of the repository, and are placed here for clarity and organization,
since they do not group well with the others.
"""
__author__ = "David Whyatt"

import numpy as np
from collections import Counter

def ratio(x: list[float], y: list[float]) -> list[float]:
    """Calculates the ratio between corresponding elements in two lists.

    Parameters
    ----------
    x : list[float]
        First list of numeric values
    y : list[float]
        Second list of numeric values. Must have same length as x.

    Returns
    -------
    list[float]
        List containing ratios of corresponding elements (x[i]/y[i]).
        Returns empty list if input lists are empty or have different lengths.
        Returns None for any division by zero.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> ratio([2, 4, 6], [1, 2, 3])  # Simple ratios
    [2.0, 2.0, 2.0]
    >>> ratio([1, 2], [0, 2])  # Division by zero
    [None, 1.0]
    >>> ratio([], [1, 2])  # Empty/unequal lists
    []
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

    Parameters
    ----------
    values : list[float]
        List of numeric values to rank
    descending : bool, optional
        If True, highest value gets rank 1.
        If False, lowest value gets rank 1.

    Returns
    -------
    list[float]
        List of ranks corresponding to the input values.
        Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> rank_values([3, 1, 4, 1, 5])  # Ascending order
    [2.0, 1.0, 3.0, 1.0, 4.0]
    >>> rank_values([3, 1, 4, 1, 5], descending=True)  # Descending order
    [3.0, 4.0, 2.0, 4.0, 1.0]
    >>> rank_values([])  # Empty input
    []
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

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    list[float]
        List with modulo 12 of each value. Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> modulo_twelve([14, 25, 36])  # Values > 12
    [2.0, 1.0, 0.0]
    >>> modulo_twelve([5, 7, 11])  # Values < 12
    [5.0, 7.0, 11.0]
    >>> modulo_twelve([])  # Empty input
    []
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

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    list[float]
        List containing absolute values of input numbers. Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> absolute_values([-1, 2, -3])  # Mixed signs
    [1.0, 2.0, 3.0]
    >>> absolute_values([0, -0.5, 1.5])  # Decimal values
    [0.0, 0.5, 1.5]
    >>> absolute_values([])  # Empty input
    []
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

    Parameters
    ----------
    values : list[float]
        List of numeric values
    x : float, optional
        Upper limit (inclusive). If None, no upper limit is applied.
    y : float, optional
        Lower limit (inclusive). If None, no lower limit is applied.

    Returns
    -------
    list[float]
        List containing only values within the specified limits. Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> limit([1, 2, 3, 4, 5], x=3)  # Upper limit only
    [1.0, 2.0, 3.0]
    >>> limit([1, 2, 3, 4, 5], y=3)  # Lower limit only
    [3.0, 4.0, 5.0]
    >>> limit([1, 2, 3, 4, 5], x=4, y=2)  # Both limits
    [2.0, 3.0, 4.0]
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

    Parameters
    ----------
    values : list[float]
        List of numeric values to analyze
    threshold : float, optional
        Minimum proportion (between 0 and 1) required for a value to be considered significant.
        Default is 0.09 (9%)

    Returns
    -------
    list[float]
        List of values that appear more than the threshold proportion of times.
        Returns empty list for empty input.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> nine_percent_significant_values([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # >9% threshold
    [1.0]
    >>> nine_percent_significant_values([1, 1, 2, 2, 3, 3], threshold=0.3)  # 30% threshold
    [1.0, 2.0, 3.0]
    >>> nine_percent_significant_values([])  # Empty input
    []
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
def circle_of_fifths(pitches: list[float], counts: list[float]) -> dict[int, float]:
    """Reorganizes two lists of pitches and counts according to the circle of fifths pattern.

    Parameters
    ----------
    pitches : list[float]
        List of pitch classes (0-12)
    counts : list[float] 
        List of counts corresponding to each pitch class

    Returns
    -------
    dict[int, float]
        Dictionary with counts mapped to pitches reordered according to circle of fifths pattern
        (0,7,2,9,4,11,6,1,8,3,10,5). Empty input returns empty dictionary.

    Raises
    ------
    ValueError
        If any pitch class is not between 0 and 12
        If input lists have different lengths

    Examples
    --------
    >>> circle_of_fifths([0, 4, 7, 9], [1, 2, 3, 2])  # Subset of pitches
    {0: 1.0, 7: 3.0, 9: 2.0, 4: 2.0}
    >>> circle_of_fifths([], [])  # Empty input
    {}
    """
    if not pitches or not counts:
        return {}

    if len(pitches) != len(counts):
        raise ValueError("Input lists must have same length")

    # Validate pitch values
    for pitch in pitches:
        if pitch < 0 or pitch > 12:
            raise ValueError("Pitch classes must be between 0 and 12")

    # Create initial dictionary from inputs
    pitch_counts = {int(p): float(c) for p, c in zip(pitches, counts)}

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

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    list[tuple[int, float]]
        List of tuples (index, value) for each extrema point.
        Returns empty list for empty input or lists shorter than 2 elements.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> contour_extrema([1, 3, 2, 4, 1])  # Simple peaks and valleys
    [(0, 1.0), (1, 3.0), (2, 2.0), (3, 4.0), (4, 1.0)]
    >>> contour_extrema([1, 2, 2, 2, 1])  # Plateau
    [(0, 1.0), (4, 1.0)]
    >>> contour_extrema([1])  # Too short
    []
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

    Parameters
    ----------
    values : list[float]
        List of numeric values

    Returns
    -------
    list[float]
        List of gradients between consecutive extrema points.
        Returns empty list for empty input or lists shorter than 2 elements.

    Raises
    ------
    TypeError
        If any element cannot be converted to float

    Examples
    --------
    >>> contour_gradients([1, 3, 2, 4, 1])  # Simple sequence
    [2.0, -1.0, 2.0, -3.0]
    >>> contour_gradients([1, 1, 1])  # Flat sequence
    [0.0]
    >>> contour_gradients([1])  # Too short
    []
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

    Parameters
    ----------
    pitch_classes : list[float]
        List of integers representing pitch classes (0-11)

    Returns
    -------
    tuple[str, float]
        Tuple containing the most likely key and its correlation value.

    Raises
    ------
    TypeError
        If pitch classes cannot be converted to integers between 0-11

    Examples
    --------
    >>> compute_tonality_vector([0, 4, 7])  # C major triad
    ('C major', 0.833...)
    >>> compute_tonality_vector([0, 4, 9])  # A minor triad
    ('a minor', 0.888...)
    >>> compute_tonality_vector([])  # Empty input
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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

        # Calculate correlations - get the off-diagonal element [0,1]
        maj_corr = np.corrcoef(pc_dist, shifted_maj)[0,1]
        min_corr = np.corrcoef(pc_dist, shifted_min)[0,1]

        correlations.append(float(maj_corr))

    # Add minor key correlations
    for i in range(12):
        shifted_min = min_vector[-i:] + min_vector[:-i]
        min_corr = np.corrcoef(pc_dist, shifted_min)[0,1]
        correlations.append(float(min_corr))

    # Create dictionary mapping key names to correlation values
    key_correlations = {}

    # Add major keys (C through B)
    major_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i, key in enumerate(major_keys):
        key_correlations[key + ' major'] = correlations[i]

    # Add minor keys (c through b)
    minor_keys = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    for i, key in enumerate(minor_keys):
        key_correlations[key + ' minor'] = correlations[i + 12]

    # Find key with maximum correlation
    max_key = max(key_correlations, key=key_correlations.get)
    return max_key, key_correlations[max_key]

def amount_of_arpeggiation(pitch_values: list[float]) -> float:
    """Calculates the proportion of notes in the melody that 
    consitute a triadic movement.

    Examines consecutive pitch intervals and counts what proportion match common
    arpeggio intervals like thirds, fifths, octaves etc.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze

    Returns
    -------
    float
        Proportion of intervals that match arpeggio patterns (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.

    Examples
    --------
    >>> amount_of_arpeggiation([60, 64, 67])  # C major triad
    1.0
    >>> amount_of_arpeggiation([60, 62, 64])  # Stepwise motion
    0.0
    >>> amount_of_arpeggiation([60])  # Single note
    0.0
    >>> amount_of_arpeggiation(None)  # Invalid input
    -1.0
    """
    if pitch_values is None:
        return -1.0

    # Calculate differences between consecutive pitches
    intervals = []
    for i in range(len(pitch_values)-1):
        interval = abs(pitch_values[i+1] - pitch_values[i])
        intervals.append(interval)

    if not intervals:
        return 0.0

    # Count occurrences of specific intervals
    target_intervals = [
        0,  # repeated notes
        3,  # minor thirds
        4,  # major thirds
        7,  # perfect fifths
        10, # minor sevenths
        11, # major sevenths
        12, # octaves
        15, # minor tenths
        16  # major tenths
    ]
    # Count how many intervals match our target arpeggio intervals
    matching_intervals = sum(1 for interval in intervals if interval in target_intervals)
    return float(matching_intervals) / len(intervals)

def chromatic_motion(pitch_values: list[float]) -> float:
    """Calculates the proportion of notes in the melody that 
    consitute a chromatic movement.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze

    Returns
    -------
    float
        Proportion of notes that move by a chromatic interval (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.

    Examples
    --------
    >>> chromatic_motion([60, 61, 62, 63, 64])  # All semitones
    1.0
    >>> chromatic_motion([60, 62, 64])  # Stepwise motion
    0.0
    >>> chromatic_motion([60, 61, 63, 64])  # Mix which includes chromatic motion
    0.666...
    >>> chromatic_motion([60])  # Single note
    0.0
    >>> chromatic_motion(None)  # Invalid input
    -1.0
    """
    if pitch_values is None:
        return -1.0

    # Calculate differences between consecutive pitches
    intervals = []
    for i in range(len(pitch_values)-1):
        interval = abs(pitch_values[i+1] - pitch_values[i])
        intervals.append(interval)

    if not intervals:
        return 0.0

    # Count how many intervals match our target chromatic intervals
    chromatic = sum(1 for interval in intervals if interval == 1)
    return float(chromatic) / len(intervals)

def stepwise_motion(pitch_values: list[float]) -> float:
    """Calculates the proportion of notes in the melody that 
    constitute a stepwise movement.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze

    Returns
    -------
    float
        Proportion of notes that move by stepwise intervals (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.

    Examples
    --------
    >>> stepwise_motion([60, 61, 62, 63, 64])  # All stepwise
    1.0
    >>> stepwise_motion([60, 63, 66])  # No stepwise motion
    0.0
    >>> stepwise_motion([60, 61, 64, 65])  # Mix which includes stepwise motion
    0.666...
    >>> stepwise_motion([60])  # Single note
    0.0
    >>> stepwise_motion(None)  # Invalid input
    -1.0
    """

    if pitch_values is None:
        return -1.0

    # Calculate differences between consecutive pitches
    intervals = []
    for i in range(len(pitch_values)-1):
        interval = abs(pitch_values[i+1] - pitch_values[i])
        intervals.append(interval)

    if not intervals:
        return 0.0

    # Count how many intervals match our target stepwise intervals
    stepwise = sum(1 for interval in intervals if interval in [1, 2])
    return float(stepwise) / len(intervals)


def repeated_notes(pitch_values: list[float]) -> float:
    """Calculates the proportion of notes in the melody that 
    are repeated.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze

    Returns
    -------
    float
        Proportion of notes that are repeated (0.0-1.0).
        Returns -1.0 if input is None.

    Examples
    --------
    >>> repeated_notes([60, 60, 62, 64])  # One note repeated 
    0.333...
    >>> repeated_notes([60, 60, 60, 60])  # All notes repeated
    1.0
    >>> repeated_notes([60, 62, 64])  # No repeated notes
    0.0
    >>> repeated_notes([60])  # Single note
    0.0
    >>> repeated_notes(None)  # Invalid input
    -1.0
    """
    value = -1.0
    if pitch_values is not None:
        intervals = [abs(pitch_values[i+1] - pitch_values[i]) 
                    for i in range(len(pitch_values)-1)]
        if intervals:
            repeated = sum(1 for interval in intervals if interval == 0)
            value = float(repeated) / len(intervals)
        else:
            value = 0.0

    return value

def melodic_embellishment(pitch_values: list[float], 
                          note_starts: list[float], 
                          note_ends: list[float]) -> float:
    
    """Calculates the proportion of notes in the melody that 
    are embellished. This is done by checking if the note is surrounded by
    notes which are at least a third the duration of the targetnote.

    Parameters
    ----------
    pitch_values : list[float]
        List of pitch values to analyze
    note_starts : list[float]
        List of note start times
    note_ends : list[float]
        List of note end times

    Returns
    -------
    float
        Proportion of notes that are embellished (0.0-1.0).
        Returns -1.0 if input is None.

    Examples
    --------
    >>> melodic_embellishment([60, 62, 64], [0, 1, 2], [1, 2, 3])  # No embellishment
    0.0
    >>> melodic_embellishment([60, 62, 64, 66], [0, 0.25, 2, 3], [0.25, 2, 2.25, 4])  # Some embellishment
    0.25...
    """
    if any([pitch_values is None, note_starts is None, note_ends is None]):
        return -1.0

    # Calculate the duration of each note
    note_durations = [note_ends[i] - note_starts[i] for i in range(len(note_starts))]
    # Count embellished notes (notes surrounded by shorter notes)
    embellished = 0
    for i in range(1, len(note_durations)-1):
        if (note_durations[i-1] < note_durations[i]/3 and
            note_durations[i+1] < note_durations[i]/3):
            embellished += 1

    # Calculate proportion of embellished notes
    if len(note_durations) > 2:  # Need at least 3 notes to have embellishment
        return float(embellished) / len(note_durations)
    return 0.0

def ukkonen_measure(pitches1: list[int], pitches2: list[int], n: int) -> int:
    """
    Calculates the Ukkonen Measure between two sequences of pitches based on n-grams.

    Parameters
    ----------
    pitches1 : list[int]
        First sequence of pitch values
    pitches2 : list[int]
        Second sequence of pitch values
    n : int
        Length of n-grams

    Returns
    -------
    int
        Ukkonen Measure value
    """
    # Generate n-grams for both pitch sequences
    s_ngrams = [tuple(pitches1[i:i+n]) for i in range(len(pitches1) - n + 1)]
    t_ngrams = [tuple(pitches2[i:i+n]) for i in range(len(pitches2) - n + 1)]

    # Count frequencies of n-grams
    s_count = Counter(s_ngrams)
    t_count = Counter(t_ngrams)

    # Get the union of n-grams
    all_ngrams = set(s_ngrams) | set(t_ngrams)

    # Calculate the Ukkonen Measure
    um = sum(abs(s_count[ngram] - t_count[ngram]) for ngram in all_ngrams)

    return um
