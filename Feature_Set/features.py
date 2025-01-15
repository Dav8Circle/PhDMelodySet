"""
This module contains functions to compute features from melodies.
Features are the product of an input list and at least one algorithm.
"""
__author__ = "David Whyatt"

from algorithms import rank_values, nine_percent_significant_values, circle_of_fifths, compute_tonality_vector
from complexity import consecutive_fifths, repetition_rate
from distributional import distribution_proportions, histogram_bins, kurtosis, skew
from interpolation_contour import InterpolationContour
from stats import range_func, standard_deviation, shannon_entropy, mode
from step_contour import StepContour
import numpy as np

# Pitch Features

def pitch_range(pitches: list[int]) -> int:
    """Calculate the range between the highest and lowest pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Range between highest and lowest pitch in semitones
    """
    return range_func(pitches)

def pitch_standard_deviation(pitches: list[int]) -> float:
    """Calculate the standard deviation of pitch values.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of pitches
    """
    return standard_deviation(pitches)

def pitch_entropy(pitches: list[int]) -> float:
    """Calculate the Shannon entropy of pitch values.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Shannon entropy of pitch distribution
    """
    return shannon_entropy(pitches)

def pcdist1(pitches: list[int], starts: list[float], ends: list[float]) -> float:
    """Calculate duration-weighted distribution of pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Duration-weighted distribution proportion of pitches
    """
    durations = [ends[i] - starts[i] for i in range(len(starts))]
    # Create weighted list by repeating each pitch according to its duration
    weighted_pitches = []
    for pitch, duration in zip(pitches, durations):
        # Convert duration to integer number of repetitions (e.g. duration 2.5 -> 25 repetitions)
        repetitions = int(duration * 10)
        weighted_pitches.extend([pitch] * repetitions)

    return distribution_proportions(weighted_pitches)

def basic_pitch_histogram(pitches: list[int]) -> dict:
    """Create histogram of pitch values within range of input pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping pitch values to counts
    """
    num_midi_notes = int(range_func(pitches))
    return histogram_bins(pitches, num_midi_notes)

def pitch_ranking(pitches: list[int]) -> float:
    """Calculate ranking of pitches in descending order.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ranking value of pitches
    """
    return rank_values(pitches, descending=True)

def melodic_pitch_variety(pitches: list[int]) -> float:
    """Calculate rate of pitch repetition.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Rate of pitch repetition
    """
    return repetition_rate(pitches)

def dominant_spread(pitches: list[int]) -> float:
    """Find longest sequence of pitch classes separated by perfect 5ths that each appear >9% of the time.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Length of longest sequence of significant pitch classes separated by perfect 5ths
    """
    pcs = [pitch % 12 for pitch in pitches]
    longest_sequence = []
    nine_percent_significant = nine_percent_significant_values(pcs)

    for i, pc in enumerate(pcs):
        if pc in nine_percent_significant:
            consecutive_fifth_pcs = consecutive_fifths(pcs[i:])
            if len(consecutive_fifth_pcs) > len(longest_sequence):
                longest_sequence = consecutive_fifth_pcs

    return int(list(longest_sequence.values())[0])

def mean_pitch(pitches: list[int]) -> float:
    """Calculate mean pitch value.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean pitch value
    """
    return np.mean(pitches)

def most_common_pitch(pitches: list[int]) -> int:
    """Find most frequently occurring pitch.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most common pitch value
    """
    return mode(pitches)

def number_of_common_pitches(pitches: list[int]) -> int:
    """Count pitch classes that appear in at least 9% of notes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of significant pitch classes
    """
    pcs = [pitch % 12 for pitch in pitches]
    significant_pcs = nine_percent_significant_values(pcs)
    return len(significant_pcs)

def number_of_pitches(pitches: list[int]) -> int:
    """Count total number of pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Total number of pitches
    """
    return len(pitches)

def folded_fifths_pitch_class_histogram(pitches: list[int]) -> dict:
    """Create histogram of pitch classes arranged in circle of fifths.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping pitch classes to counts, arranged by circle of fifths
    """
    # Get pitch classes and count occurrences
    pcs = [pitch % 12 for pitch in pitches]
    # Count occurrences of each pitch class
    unique = []
    counts = []
    for pc in set(pcs):
        unique.append(pc)
        counts.append(pcs.count(pc))
    return circle_of_fifths(unique, counts)

def pitch_class_kurtosis_after_folding(pitches: list[int]) -> float:
    """Calculate kurtosis of folded fifths pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Kurtosis of folded fifths histogram values, or 0 for empty input
    """
    pitches = [pitch % 12 for pitch in pitches]
    histogram = folded_fifths_pitch_class_histogram(pitches)
    if not histogram:
        return 0.0
    return kurtosis(list(histogram.keys()))

def pitch_class_skewness_after_folding(pitches: list[int]) -> float:
    """Calculate skewness of folded fifths pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Skewness of folded fifths histogram values, or 0 for empty input
    """
    pitches = [pitch % 12 for pitch in pitches]
    histogram = folded_fifths_pitch_class_histogram(pitches)
    if not histogram:
        return 0.0
    return skew(list(histogram.keys()))

def pitch_class_variability_after_folding(pitches: list[int]) -> float:
    """Calculate standard deviation of folded fifths pitch class histogram.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of folded fifths histogram values, or 0 for empty input
    """
    pitches = [pitch % 12 for pitch in pitches]
    histogram = folded_fifths_pitch_class_histogram(pitches)
    if not histogram:
        return 0.0
    return standard_deviation(list(histogram.keys()))

# Interval Features

def pitch_interval(pitches: list[int]) -> list[int]:
    """Calculate intervals between consecutive pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    list[int]
        List of intervals between consecutive pitches in semitones
    """
    return [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]

def absolute_interval_range(pitches: list[int]) -> int:
    """Calculate range between largest and smallest absolute interval size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Range between largest and smallest absolute interval in semitones
    """
    return range_func([abs(x) for x in pitch_interval(pitches)])

def mean_absolute_interval(pitches: list[int]) -> float:
    """Calculate mean absolute interval size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Mean absolute interval size in semitones
    """
    return np.mean([abs(x) for x in pitch_interval(pitches)])

# Alias for mean_absolute_interval / FANTASTIC vs jSymbolic
mean_melodic_interval = mean_absolute_interval

def standard_deviation_absolute_interval(pitches: list[int]) -> float:
    """Calculate standard deviation of absolute interval sizes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Standard deviation of absolute interval sizes in semitones
    """
    return np.std([abs(x) for x in pitch_interval(pitches)])

def modal_interval(pitches: list[int]) -> int:
    """Find most common interval size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Most frequent interval size in semitones
    """
    return mode(pitch_interval(pitches))

# Alias for modal_interval / FANTASTIC vs jSymbolic
most_common_interval = modal_interval

def interval_entropy(pitches: list[int]) -> float:
    """Calculate Shannon entropy of interval distribution.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Shannon entropy of interval sizes
    """
    return shannon_entropy(pitch_interval(pitches))

def ivdist1(pitches: list[int], starts: list[float], ends: list[float]) -> float:
    """Calculate duration-weighted distribution of intervals.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Duration-weighted distribution proportion of intervals
    """
    intervals = pitch_interval(pitches)
    durations = [ends[i] - starts[i] for i in range(len(starts))]
    
    weighted_intervals = []
    for interval, duration in zip(intervals, durations[:-1]):
        repetitions = int(duration * 10)
        weighted_intervals.extend([interval] * repetitions)

    return distribution_proportions(weighted_intervals)

def interval_direction(pitches: list[int]) -> list[str]:
    """Determine direction of each interval.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    list[str]
        List of interval directions ("up", "same", or "down")
    """
    return ["up" if pitches[i + 1] > pitches[i]
            else "same" if pitches[i + 1] == pitches[i]
            else "down"
            for i in range(len(pitches) - 1)]

def average_interval_span_by_melodic_arcs(pitches: list[int]) -> float:
    """Calculate average interval span of melodic arcs.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Average interval span of melodic arcs, or 0.0 if no arcs found
    """
    total_intervals = 0
    number_intervals = 0

    intervals = pitch_interval(pitches)
    direction = 0
    interval_so_far = 0

    for interval in intervals:
        if direction == -1:  # Arc is currently descending
            if interval < 0:
                interval_so_far += abs(interval)
            elif interval > 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = 1

        elif direction == 1:  # Arc is currently ascending
            if interval > 0:
                interval_so_far += abs(interval)
            elif interval < 0:
                total_intervals += interval_so_far
                number_intervals += 1
                interval_so_far = abs(interval)
                direction = -1

        elif direction == 0:  # Arc is currently stationary
            if interval > 0:
                direction = 1
                interval_so_far += abs(interval)
            elif interval < 0:
                direction = -1
                interval_so_far += abs(interval)

    if number_intervals == 0:
        value = 0.0
    else:
        value = total_intervals / number_intervals

    return value

def distance_between_most_prevalent_melodic_intervals(pitches: list[int]) -> float:
    """Calculate absolute difference between two most common interval sizes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Absolute difference between two most common intervals, or 0.0 if fewer than 2 intervals
    """
    if len(pitches) < 2:
        return 0.0

    intervals = pitch_interval(pitches)

    interval_counts = {}
    for interval in intervals:
        interval_counts[interval] = interval_counts.get(interval, 0) + 1

    if len(interval_counts) < 2:
        return 0.0

    sorted_intervals = sorted(interval_counts.items(), key=lambda x: x[1], reverse=True)
    most_common = sorted_intervals[0][0]
    second_most_common = sorted_intervals[1][0]
    return abs(most_common - second_most_common)

def melodic_interval_histogram(pitches: list[int]) -> dict:
    """Create histogram of interval sizes.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    dict
        Dictionary mapping interval sizes to counts
    """
    intervals = pitch_interval(pitches)
    num_intervals = int(range_func(intervals))
    return histogram_bins(intervals, num_intervals)

def melodic_large_intervals(pitches: list[int]) -> float:
    """Calculate proportion of intervals >= 13 semitones.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of large intervals, or -1.0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return -1.0
    large_intervals = sum(1 for interval in intervals if abs(interval) >= 13)
    return large_intervals / len(intervals) if intervals else 0.0

def variable_melodic_intervals(pitches: list[int], interval_level: int) -> float:
    """Calculate proportion of intervals >= specified size.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    interval_level : int
        Minimum interval size in semitones

    Returns
    -------
    float
        Proportion of intervals >= interval_level, or -1.0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return -1.0
    large_intervals = sum(1 for interval in intervals if abs(interval) >= interval_level)
    return large_intervals / len(intervals) if intervals else 0.0

def number_of_common_melodic_intervals(pitches: list[int]) -> int:
    """Count intervals that appear in at least 9% of melodic transitions.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of significant intervals
    """
    if len(pitches) < 2:
        return 0
        
    intervals = pitch_interval(pitches)
    significant_intervals = nine_percent_significant_values(intervals)
    
    return len(significant_intervals)

def prevalence_of_most_common_melodic_interval(pitches: list[int]) -> float:
    """Count occurrences of most common interval.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Count of most common interval, or 0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return 0
    
    interval_counts = {}
    for interval in intervals:
        interval_counts[interval] = interval_counts.get(interval, 0) + 1
        
    return max(interval_counts.values())

# Contour Features
def get_step_contour_features(pitches: list[int], starts: list[float], ends: list[float]) -> StepContour:
    """Calculate step contour features.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    StepContour
        StepContour object with global variation, direction and local variation
    """
    durations = [ends[i] - starts[i] for i in range(len(starts))]
    sc = StepContour(pitches, durations)
    return sc.global_variation, sc.global_direction, sc.local_variation

def get_interpolation_contour_features(pitches: list[int], starts: list[float]) -> InterpolationContour:
    """Calculate interpolation contour features.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values
    starts : list[float]
        List of note start times

    Returns
    -------
    InterpolationContour
        InterpolationContour object with direction, gradient and class features
    """
    ic = InterpolationContour(pitches, starts)
    return (ic.global_direction, ic.mean_gradient, ic.gradient_std,
            ic.direction_changes, ic.class_label)

# Duration Features

def duration_range(starts: list[float], ends: list[float]) -> float:
    """Calculate range between longest and shortest note duration.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Range between longest and shortest duration
    """
    return range_func([ends[i] - starts[i] for i in range(len(starts))])

def modal_duration(starts: list[float], ends: list[float]) -> float:
    """Find most common note duration.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Most frequent note duration
    """
    durations = [ends[i] - starts[i] for i in range(len(starts))]
    return mode(durations)

def duration_entropy(starts: list[float], ends: list[float]) -> float:
    """Calculate Shannon entropy of duration distribution.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Shannon entropy of note durations
    """
    durations = [ends[i] - starts[i] for i in range(len(starts))]
    return shannon_entropy(durations)

def length(starts: list[float]) -> float:
    """Count total number of notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Total number of notes
    """
    return len(starts)

def global_duration(starts: list[float], ends: list[float]) -> float:
    """Calculate total duration from first note start to last note end.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Total duration of melody
    """
    return ends[-1] - starts[0]

def note_density(starts: list[float], ends: list[float]) -> float:
    """Calculate average number of notes per unit time.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Note density (notes per unit time)
    """
    return len(starts) / global_duration(starts, ends)

def ioi(starts: list[float]) -> list[float]:
    """Calculate inter-onset intervals between consecutive notes.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[float]
        List of inter-onset intervals
    """
    return [starts[i+1] - starts[i] for i in range(len(starts)-1)]

def ioi_ratio(starts: list[float]) -> list[float]:
    """Calculate ratios between consecutive inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[float]
        List of IOI ratios
    """
    ioi_list = ioi(starts)
    return [ioi_list[i+1]/ioi_list[i] for i in range(len(ioi_list)-1)]

def ioi_contour(starts: list[float]) -> list[int]:
    """Calculate contour of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    list[int]
        List of contour values (-1: shorter, 0: same, 1: longer)
    """
    ioi_list = ioi(starts)
    if len(ioi_list) < 2:
        return []

    ratios = [ioi_list[i+1]/ioi_list[i] for i in range(len(ioi_list)-1)]

    return [int(np.sign(ratio - 1)) for ratio in ratios]

# Tonality Features
def tonalness(pitches: list[int]) -> float:
    """Calculate tonalness as magnitude of highest key correlation.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Magnitude of highest key correlation value
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    _, correlation = compute_tonality_vector(pitch_classes)
    return abs(correlation)

def tonal_clarity(pitches: list[int]) -> float:
    """Calculate ratio between top two key correlation values.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio between highest and second highest key correlation values.
        Returns 1.0 if fewer than 2 correlation values.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    
    if len(correlations) < 2:
        return -1.0
        
    # Get top 2 correlation values
    top_corr = abs(correlations[0][1])
    second_corr = abs(correlations[1][1])
    
    # Avoid division by zero
    if second_corr == 0:
        return 1.0
        
    return top_corr / second_corr

def tonal_spike(pitches: list[int]) -> float:
    """Calculate ratio between highest key correlation and sum of all others.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Ratio between highest correlation value and sum of all others.
        Returns 1.0 if fewer than 2 correlation values or sum is zero.
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    
    if len(correlations) < 2:
        return -1.0
        
    # Get highest correlation and sum of rest
    top_corr = abs(correlations[0][1])
    other_sum = sum(abs(corr[1]) for corr in correlations[1:])
    
    # Avoid division by zero
    if other_sum == 0:
        return 1.0
        
    return top_corr / other_sum

def get_key_distances() -> dict[str, int]:
    """Returns a dictionary mapping key names to their semitone distances from C.
    
    Returns
    -------
    dict[str, int]
        Dictionary mapping key names (both major and minor) to semitone distances from C.
    """
    return {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
        'c': 0, 'c#': 1, 'd': 2, 'd#': 3, 'e': 4, 'f': 5,
        'f#': 6, 'g': 7, 'g#': 8, 'a': 9, 'a#': 10, 'b': 11
    }

def referent(pitches: list[int]) -> int:
    '''
    Feature that describes the chromatic interval of the key centre from C.
    '''
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)

    if not correlations:
        return -1

    # Get the key name from the highest correlation
    key_name = correlations[0][0].split()[0]  # Take first word (key name without major/minor)

    # Map key names to semitone distances from C


    key_distances = get_key_distances()

    return key_distances[key_name]

def inscale(pitches: list[int]) -> int:
    '''
    Captures whether the melody contains any notes which deviate from the estimated key.
    '''
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)[0]
    key_centre = correlations[0]
    print(key_centre)

    # Get major/minor scales based on key
    if 'major' in key_centre:
        # Major scale pattern: W-W-H-W-W-W-H (W=2 semitones, H=1 semitone)
        scale = [0, 2, 4, 5, 7, 9, 11]
    else:
        # Natural minor scale pattern: W-H-W-W-H-W-W
        scale = [0, 2, 3, 5, 7, 8, 10]

    # Get key root pitch class
    key_name = key_centre.split()[0]
    key_distances = get_key_distances()
    root = key_distances[key_name]

    # Transpose scale to key
    scale = [(note + root) % 12 for note in scale]

    # Check if any pitch classes are outside the scale
    for pc in pitch_classes:
        if pc not in scale:
            return 0

    return 1
