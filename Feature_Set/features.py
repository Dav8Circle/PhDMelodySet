"""
This module contains functions to compute features from melodies.
Features are the product of an input list and at least one algorithm.
"""
__author__ = "David Whyatt"

import json
import math
import csv
from collections import Counter
from random import choices
import time
from typing import Dict
from multiprocessing import Pool, cpu_count
from algorithms import (
    rank_values, nine_percent_significant_values, circle_of_fifths,
    compute_tonality_vector, arpeggiation_proportion,
    chromatic_motion_proportion, stepwise_motion_proportion,
    repeated_notes_proportion, melodic_embellishment_proportion,
    longest_monotonic_conjunct_scalar_passage, longest_conjunct_scalar_passage,
    proportion_conjunct_scalar, proportion_scalar
)
from complexity import (
    consecutive_fifths, repetition_rate, yules_k, simpsons_d, sichels_s, honores_h, mean_entropy,
    mean_productivity
)
from distributional import distribution_proportions, histogram_bins, kurtosis, skew
from interpolation_contour import InterpolationContour
from mtypes import FantasticTokenizer
from narmour import (
    proximity, closure, registral_direction, registral_return, intervallic_difference)
from representations import Melody
from stats import range_func, standard_deviation, shannon_entropy, mode
from step_contour import StepContour
import numpy as np
import scipy
from tqdm import tqdm

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
    return int(range_func(pitches))

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
    return float(standard_deviation(pitches))

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
    return float(shannon_entropy(pitches))

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

    return len(longest_sequence)

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
    return int(mode(pitches))

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
    return int(len(significant_pcs))

def number_of_pitches(pitches: list[int]) -> int:
    """Count number of unique pitches.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    int
        Number of unique pitches
    """
    return int(len(set(pitches)))

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
    return float(kurtosis(list(histogram.keys())))

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
    return float(skew(list(histogram.keys())))

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
    return float(standard_deviation(list(histogram.keys())))

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
    return int(range_func([abs(x) for x in pitch_interval(pitches)]))

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
    return float(np.mean([abs(x) for x in pitch_interval(pitches)]))

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
    return float(np.std([abs(x) for x in pitch_interval(pitches)]))

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
    return int(mode(pitch_interval(pitches)))

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
    return float(shannon_entropy(pitch_interval(pitches)))

def ivdist1(pitches: list[int], starts: list[float], ends: list[float]) -> dict:
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

def interval_direction(pitches: list[int]) -> tuple[float, float]:
    """Determine direction of each interval and calculate mean and standard deviation.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of interval directions, where:
        1 represents upward motion
        0 represents same pitch
        -1 represents downward motion
    """
    directions = [1 if pitches[i + 1] > pitches[i]
                 else 0 if pitches[i + 1] == pitches[i]
                 else -1
            for i in range(len(pitches) - 1)]
    
    if not directions:
        return 0.0, 0.0
        
    mean = sum(directions) / len(directions)
    variance = sum((x - mean) ** 2 for x in directions) / len(directions)
    std_dev = math.sqrt(variance)

    return float(mean), float(std_dev)

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

    return float(value)

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
    return float(abs(most_common - second_most_common))

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
    return float(large_intervals / len(intervals) if intervals else 0.0)

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
    return float(large_intervals / len(intervals) if intervals else 0.0)

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
    
    return int(len(significant_intervals))

def prevalence_of_most_common_melodic_interval(pitches: list[int]) -> float:
    """Calculate proportion of most common interval.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of most common interval, or 0 if no intervals
    """
    intervals = pitch_interval(pitches)
    if not intervals:
        return 0
    
    interval_counts = {}
    for interval in intervals:
        interval_counts[interval] = interval_counts.get(interval, 0) + 1
        
    return float(max(interval_counts.values()) / len(intervals))

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

def get_tempo(melody: Melody) -> float:
    """Access tempo of melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze

    Returns
    -------
    float
        Tempo of melody in bpm
    
    """
    return melody.tempo

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

def mean_duration(starts: list[float], ends: list[float]) -> float:
    """Calculate mean note duration.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Mean note duration
    """
    durations = [ends[i] - starts[i] for i in range(len(starts))]
    return float(np.mean(durations))

def duration_standard_deviation(starts: list[float], ends: list[float]) -> float:
    """Calculate standard deviation of note durations.

    Parameters
    ---------- 
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    float
        Standard deviation of note durations
    """
    durations = [ends[i] - starts[i] for i in range(len(starts))]
    return float(np.std(durations))

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

def number_of_durations(starts: list[float], ends: list[float]) -> int:
    """Count number of unique note durations.

    Parameters
    ----------
    starts : list[float]
        List of note start times    
    ends : list[float]
        List of note end times

    Returns
    -------
    int
        Number of unique note durations
    """
    durations = [ends[i] - starts[i] for i in range(len(starts))]
    return len(set(durations))

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

def ioi(starts: list[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    if not intervals:
        return 0.0, 0.0
    return float(np.mean(intervals)), float(np.std(intervals))

def ioi_ratio(starts: list[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation of inter-onset interval ratios.
    
    Parameters
    ----------
    starts : list[float]
        List of note start times
        
    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of IOI ratios
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    if len(intervals) < 2:
        return 0.0, 0.0

    ratios = [intervals[i]/intervals[i-1] for i in range(1, len(intervals))]
    return float(np.mean(ratios)), float(np.std(ratios))

def ioi_range(starts: list[float]) -> float:
    """Calculate range of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Range of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    return max(intervals) - min(intervals)

def ioi_mean(starts: list[float]) -> float:
    """Calculate mean of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Mean of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    return float(np.mean(intervals))

def ioi_standard_deviation(starts: list[float]) -> float:
    """Calculate standard deviation of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    float
        Standard deviation of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    return float(np.std(intervals))

def ioi_contour(starts: list[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation of IOI contour.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of contour values (-1: shorter, 0: same, 1: longer)
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    if len(intervals) < 2:
        return 0.0, 0.0
        
    ratios = [intervals[i]/intervals[i-1] for i in range(1, len(intervals))]
    contour = [int(np.sign(ratio - 1)) for ratio in ratios]
    return float(np.mean(contour)), float(np.std(contour))

def duration_histogram(starts: list[float], ends: list[float]) -> dict:
    """Calculate histogram of note durations.

    Parameters
    ----------
    starts : list[float]
        List of note start times
    ends : list[float]
        List of note end times

    Returns
    -------
    dict
        Histogram of note durations
    """
    durations = [ends[i] - starts[i] for i in range(len(starts))]
    num_durations = len(set(durations))
    return histogram_bins(durations, num_durations)

def ioi_histogram(starts: list[float]) -> dict:
    """Calculate histogram of inter-onset intervals.

    Parameters
    ----------
    starts : list[float]
        List of note start times

    Returns
    -------
    dict
        Histogram of inter-onset intervals
    """
    intervals = [starts[i] - starts[i-1] for i in range(1, len(starts))]
    num_intervals = len(set(intervals))
    return histogram_bins(intervals, num_intervals)

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
    correlation = compute_tonality_vector(pitch_classes)
    return correlation[0][1]

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

def tonal_entropy(pitches: list[int]) -> float:
    """Calculate tonal entropy as the entropy across the key correlations.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Entropy of the tonality vector correlation distribution
    """
    pitch_classes = [pitch % 12 for pitch in pitches]
    correlations = compute_tonality_vector(pitch_classes)
    if not correlations:
        return -1.0

    # Calculate entropy of correlation distribution
    # Extract just the correlation values and normalize them to positive values
    corr_values = [abs(corr[1]) for corr in correlations]

    # Calculate entropy of the correlation distribution
    return shannon_entropy(corr_values)

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

def temperley_likelihood(pitches: list[int]) -> float:
    '''
    Calculates the likelihood of a melody using Bayesian reasoning,
    according to David Temperley's model 
    (http://davidtemperley.com/wp-content/uploads/2015/11/temperley-cs08.pdf).
    '''
    # represent all possible notes as int
    notes_ints = np.arange(0, 120, 1)

    # Calculate central pitch profile
    central_pitch_profile = scipy.stats.norm.pdf(notes_ints, loc=68, scale=np.sqrt(5.0))
    central_pitch = choices(notes_ints, central_pitch_profile)
    range_profile = scipy.stats.norm.pdf(notes_ints, loc=central_pitch, scale=np.sqrt(23.0))

    # Get key probabilities

    rpk_major = [0.184, 0.001, 0.155, 0.003, 0.191, 0.109, 0.005, 0.214, 0.001, 0.078, 0.004, 0.055] * 10
    rpk_minor = [0.192, 0.005, 0.149, 0.179, 0.002, 0.144, 0.002, 0.201, 0.038, 0.012, 0.053, 0.022] * 10

    # Calculate total probability
    total_prob = 1.0
    for i in range(1, len(pitches)):
        # Calculate proximity profile centered on previous note
        prox_profile = scipy.stats.norm.pdf(notes_ints, loc=pitches[i-1], scale=np.sqrt(10))
        rp = range_profile * prox_profile

        # Apply key profile based on major/minor
        if 'major' in compute_tonality_vector([p % 12 for p in pitches])[0][0]:
            rpk = rp * rpk_major
        else:
            rpk = rp * rpk_minor

        # Normalize probabilities
        rpk_normed = rpk / np.sum(rpk)

        # Get probability of current note
        note_prob = rpk_normed[pitches[i]]
        total_prob *= note_prob

    return total_prob

def tonalness_histogram(pitches: list[int]) -> dict:
    '''
    Calculates the histogram of KS correlation values.
    '''
    p = [p % 12 for p in pitches]
    return histogram_bins(compute_tonality_vector(p)[0][1], 24)

def get_narmour_features(melody: Melody) -> Dict:
    """Calculate Narmour's implication-realization features.

    Parameters
    ----------
    melody : Melody
        The melody to analyze as a Melody object

    Returns
    -------
    Dict
        Dictionary containing scores for:
        - Registral direction (0 or 1)
        - Proximity (0-6)
        - Closure (0-2)
        - Registral return (0-3)
        - Intervallic difference (0 or 1)

    Notes
    -----
    Features represent:
    - Registral direction: Large intervals followed by direction change
    - Proximity: Closeness of consecutive pitches
    - Closure: Direction changes and interval size changes
    - Registral return: Return to previous pitch level
    - Intervallic difference: Relationship between consecutive intervals
    """
    pitches = melody.pitches
    return {
        'registral_direction': registral_direction(pitches),
        'proximity': proximity(pitches),
        'closure': closure(pitches),
        'registral_return': registral_return(pitches),
        'intervallic_difference': intervallic_difference(pitches)
    }

# Melodic Movement Features
def amount_of_arpeggiation(pitches: list[int]) -> float:
    """Calculate the proportion of notes in the melody that constitute triadic movement.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that match arpeggio patterns (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return arpeggiation_proportion(pitches)


def chromatic_motion(pitches: list[int]) -> float:
    """Calculate the proportion of chromatic motion in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are chromatic (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return chromatic_motion_proportion(pitches)

def melodic_embellishment(pitches: list[int], starts: list[float], ends: list[float]) -> float:
    """Calculate proportion of melodic embellishments (e.g. trills, turns, neighbor tones).

    Melodic embellishments are identified by looking for notes with a duration 1/3rd of the
    adjacent note's duration that move away from and return to a pitch level, or oscillate
    between two pitches.
    

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
        Proportion of intervals that are embellishments (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return melodic_embellishment_proportion(pitches, starts, ends)

def repeated_notes(pitches: list[int]) -> float:
    """Calculate the proportion of repeated notes in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are repeated notes (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return repeated_notes_proportion(pitches)

def stepwise_motion(pitches: list[int]) -> float:
    """Calculate the proportion of stepwise motion in the melody.

    Parameters
    ----------
    pitches : list[int]
        List of MIDI pitch values

    Returns
    -------
    float
        Proportion of intervals that are stepwise (0.0-1.0).
        Returns -1.0 if input is None, 0.0 if input is empty or has only one value.
    """
    return stepwise_motion_proportion(pitches)

def get_mtype_features(melody: Melody) -> dict:
    """Calculate various n-gram statistics for the melody.

    Parameters
    ----------
    melody : Melody
        The melody to analyze as a Melody object

    Returns
    -------
    dict
        Dictionary containing complexity measures averaged across n-gram lengths
    """
    pitches = melody.pitches
    starts = melody.starts
    ends = melody.ends
    tokenizer = FantasticTokenizer()
    # We don't actually use tokens, but it initializes self.phrases
    tokens = tokenizer.tokenize_melody(pitches, starts, ends)

    # Get counts for each n-gram length
    ngram_counts = []
    for n in range(1, 5):
        counts = tokenizer.ngram_counts(n=n)
        ngram_counts.append(counts)

    # Calculate complexity measures
    return {
        'yules_k': yules_k(ngram_counts),
        'simpsons_d': simpsons_d(ngram_counts),
        'sichels_s': sichels_s(ngram_counts),
        'honores_h': honores_h(ngram_counts),
        'mean_entropy': mean_entropy(ngram_counts),
        'mean_productivity': mean_productivity(ngram_counts)
    }

def get_ngram_document_frequency(ngram: tuple, corpus_stats: dict) -> int:
    """Retrieve the document frequency for a given n-gram from the corpus statistics.
    
    Parameters
    ----------
    ngram : tuple
        The n-gram to look up
    corpus_stats : dict
        Dictionary containing corpus statistics
        
    Returns
    -------
    int
        Document frequency count for the n-gram
    """
    ngram_str = str(ngram)
    # Get document frequencies dictionary
    doc_freqs = corpus_stats.get('document_frequencies', {})
    
    # Look up the count for this ngram
    if ngram_str in doc_freqs:
        return doc_freqs[ngram_str].get('count', 0)
    return 0

def compute_tfdf_spearman(melody: Melody) -> float:
    """Compute Spearman correlation between term and document frequencies.
    
    This follows FANTASTIC's implementation where:
    - TF is the frequency of an n-gram in the melody 
    - DF is the number of melodies in the corpus containing the n-gram
    - Spearman correlation is calculated between TF and DF values
    - For ties in TF/DF values, the maximum rank is assigned
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Spearman correlation coefficient between -1 and 1, or 0.0 if insufficient n-grams
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)

    tf_values = []
    df_values = []
    
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    # Get TF and DF values for each n-gram length
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        for ngram, tf in ngram_counts.items():
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                tf_values.append(tf)
                df_values.append(df)

    # Need at least 2 points for correlation
    if len(tf_values) < 2:
        return 0.0

    # Calculate Spearman correlation with maximum rank for ties
    if len(tf_values) > 0 and len(df_values) > 0:
        correlation = scipy.stats.spearmanr(tf_values, df_values)[0]
        return float(correlation) if not math.isnan(correlation) else 0.0
    else:
        return 0.0

def compute_tfdf_kendall(melody: Melody) -> float:
    """Compute Kendall correlation between term and document frequencies.
    
    This follows FANTASTIC's implementation where:
    - TF is the frequency of an n-gram in the melody 
    - DF is the number of melodies in the corpus containing the n-gram
    - Kendall correlation is calculated between TF and DF values
    - For ties in TF/DF values, the minimum rank is assigned
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Kendall correlation coefficient between -1 and 1, or 0.0 if insufficient n-grams
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)

    tf_values = []
    df_values = []
    
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    # Get TF and DF values for each n-gram length
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        for ngram, tf in ngram_counts.items():
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                tf_values.append(tf)
                df_values.append(df)

    # Need at least 2 points for correlation
    if len(tf_values) < 2:
        return 0.0

    # Calculate Kendall correlation with minimum rank for ties
    if len(tf_values) > 0 and len(df_values) > 0:
        correlation = scipy.stats.kendalltau(tf_values, df_values)[0]
        return float(correlation) if not math.isnan(correlation) else 0.0
    else:
        return 0.0

def compute_tfdf(melody: Melody) -> float:
    """Compute mean log TFDF (Term Frequency-Document Frequency) for a melody.
    
    This follows FANTASTIC's implementation where:
    - TF is the frequency of an n-gram in the melody
    - DF is the number of melodies in the corpus containing the n-gram
    - TFDF is calculated for each n-gram and averaged
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Mean log TFDF score, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)

    tfdf_values = []
    # Load corpus statistics from JSON file
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    # Calculate TFDF using dot product for each n-gram length
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        if not ngram_counts:
            continue
            
        # Get TF and DF vectors
        tf_vector = []
        df_vector = []
        for ngram, tf in ngram_counts.items():
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                # Log transform the frequencies
                tf_vector.append(math.log(tf + 1))
                df_vector.append(math.log(df + 1))
        
        # Calculate dot product if vectors are non-empty
        if tf_vector and df_vector:
            # Convert to numpy arrays for dot product
            tf_array = np.array(tf_vector)
            df_array = np.array(df_vector)
            
            # Normalize vectors
            tf_norm = tf_array / np.sqrt(np.sum(tf_array**2))
            df_norm = df_array / np.sqrt(np.sum(df_array**2))
            
            # Calculate dot product and append
            tfdf = np.dot(tf_norm, df_norm)
            tfdf_values.append(tfdf)

    # Calculate mean TFDF
    if tfdf_values:
        mean_tfdf = float(np.mean(tfdf_values))
    else:
        mean_tfdf = 0.0

    return mean_tfdf

def compute_norm_log_dist(melody: Melody) -> float:
    """Compute normalized distance between term and document frequencies.
    
    This follows FANTASTIC's implementation where:
    - TF is the frequency of an n-gram in the melody
    - DF is the number of melodies in the corpus containing the n-gram
    - Instead of multiplying TF*DF, takes difference between normalized vectors
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Normalized log distance score, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)

    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    distances = []
    # Calculate distances for each n-gram length
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        if not ngram_counts:
            continue
            
        # Get total term frequency for normalization
        total_tf = sum(ngram_counts.values())
        
        for ngram, tf in ngram_counts.items():
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                # Normalize frequencies by total counts
                norm_tf = tf / total_tf
                norm_df = df / len(corpus_stats['document_frequencies'])
                
                # Calculate absolute difference
                dist = abs(norm_tf - norm_df)
                distances.append(dist)

    # Calculate mean normalized distance
    if distances:
        mean_dist = float(np.mean(distances))
    else:
        mean_dist = 0.0

    return mean_dist

def compute_max_log_df(melody: Melody) -> float:
    """Compute maximum document frequency feature (mtcf.max.log.DF).
    
    This is the logarithm of the m-type contained in the analysis that occurs 
    in the maximum number of melodies in the corpus.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Maximum log document frequency, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)

    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    max_df = 0
    # Find maximum document frequency across all n-gram lengths
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        for ngram in ngram_counts:
            df = get_ngram_document_frequency(ngram, corpus_stats)
            max_df = max(max_df, df)

    # Take log of maximum document frequency
    if max_df > 0:
        return math.log(max_df)
    return 0.0

def compute_min_log_df(melody: Melody) -> float:
    """Compute minimum document frequency feature (mtcf.min.log.DF).
    
    This is the logarithm of the m-type contained in the analysis that occurs 
    in the minimum number of melodies in the corpus.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Minimum log document frequency, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    min_df = float('inf')  # Initialize to infinity
    # Find minimum document frequency across all n-gram lengths
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        for ngram in ngram_counts:
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:  # Only consider non-zero frequencies
                min_df = min(min_df, df)

    # Take log of minimum document frequency
    if min_df < float('inf'):
        return math.log(min_df)
    return 0.0

def compute_mean_log_df(melody: Melody) -> float:
    """Compute mean document frequency feature (mtcf.mean.log.DF).
    
    This is the mean logarithm of the document frequencies of all m-types
    contained in the analysis.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Mean log document frequency, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    total_log_df = 0.0
    count = 0
    # Calculate sum of log document frequencies across all n-gram lengths
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        for ngram in ngram_counts:
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:  # Only consider non-zero frequencies
                total_log_df += math.log(df)
                count += 1

    # Calculate mean log document frequency
    if count > 0:
        return total_log_df / count
    return 0.0

def compute_mean_df_entropy(melody: Melody) -> float:
    """Compute mean document frequency entropy feature (mtcf.mean.entropy).
    
    This is similar to mean.entropy but uses document frequencies from the corpus
    instead of frequencies within the analysis melody. The entropy values are
    averaged over the various m-type lengths.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Mean document frequency entropy, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
    
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    total_entropy = 0.0
    n_lengths = 0
    
    # Calculate entropy for each n-gram length
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        if not ngram_counts:
            continue
            
        # Get document frequencies for this n-gram length
        dfs = []
        total_df = 0
        for ngram in ngram_counts:
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                dfs.append(df)
                total_df += df
                
        if dfs and total_df > 0:
            # Calculate probabilities
            probs = [df/total_df for df in dfs]
            # Use shannon_entropy function
            entropy = shannon_entropy(probs)
            total_entropy += entropy
            n_lengths += 1
    
    # Return average entropy across n-gram lengths
    if n_lengths > 0:
        return total_entropy / n_lengths
    return 0.0

def compute_mean_df_productivity(melody: Melody) -> float:
    """Compute mean document frequency productivity for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Mean document frequency productivity, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
    
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    ngram_dfs = []
    # Calculate document frequencies for each n-gram length
    for n in range(1, 5):
        df_counts = Counter()
        ngram_counts = tokenizer.ngram_counts(n=n)
        
        if not ngram_counts:
            continue
            
        # Get document frequencies for this n-gram length
        for ngram in ngram_counts:
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                df_counts[df] += 1
                
        if df_counts:
            ngram_dfs.append(df_counts)
    
    # Use mean_productivity function to calculate result
    return mean_productivity(ngram_dfs)

def compute_mean_df_yules_k(melody: Melody) -> float:
    """Compute mean document frequency Yule's K for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Mean document frequency Yule's K, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
    
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    ngram_dfs = []
    # Calculate document frequencies for each n-gram length
    for n in range(1, 5):
        df_counts = Counter()
        ngram_counts = tokenizer.ngram_counts(n=n)
        
        if not ngram_counts:
            continue
            
        # Get document frequencies for this n-gram length
        for ngram in ngram_counts:
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                df_counts[df] += 1
                
        if df_counts:
            ngram_dfs.append(df_counts)
    
    # Use yules_k function to calculate result
    return yules_k(ngram_dfs)

def compute_mean_df_simpsons_d(melody: Melody) -> float:
    """Compute mean document frequency Simpson's D for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Mean document frequency Simpson's D, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
    
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    ngram_dfs = []
    # Calculate document frequencies for each n-gram length
    for n in range(1, 5):
        df_counts = Counter()
        ngram_counts = tokenizer.ngram_counts(n=n)
        
        if not ngram_counts:
            continue
            
        # Get document frequencies for this n-gram length
        for ngram in ngram_counts:
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                df_counts[df] += 1
                
        if df_counts:
            ngram_dfs.append(df_counts)
    
    # Use simpsons_d function to calculate result
    return simpsons_d(ngram_dfs)

def compute_mean_df_sichels_s(melody: Melody) -> float:
    """Compute mean document frequency Sichel's S for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Mean document frequency Sichel's S, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
    
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    ngram_dfs = []
    # Calculate document frequencies for each n-gram length
    for n in range(1, 5):
        df_counts = Counter()
        ngram_counts = tokenizer.ngram_counts(n=n)
        
        if not ngram_counts:
            continue
            
        # Get document frequencies for this n-gram length
        for ngram in ngram_counts:
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                df_counts[df] += 1
                
        if df_counts:
            ngram_dfs.append(df_counts)
    
    # Use sichels_s function to calculate result
    return sichels_s(ngram_dfs)

def compute_mean_df_honores_h(melody: Melody) -> float:
    """Compute mean document frequency Honoré's H for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Mean document frequency Honoré's H, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
    
    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    ngram_dfs = []
    # Calculate document frequencies for each n-gram length
    for n in range(1, 5):
        df_counts = Counter()
        ngram_counts = tokenizer.ngram_counts(n=n)
        
        if not ngram_counts:
            continue
            
        # Get document frequencies for this n-gram length
        for ngram in ngram_counts:
            df = get_ngram_document_frequency(ngram, corpus_stats)
            if df > 0:
                df_counts[df] += 1
                
        if df_counts:
            ngram_dfs.append(df_counts)
    
    # Use honores_h function to calculate result
    return honores_h(ngram_dfs)

def compute_mean_global_weight(melody: Melody) -> float:
    """Compute mean global weight (glob.w) for a melody.
    
    This follows Quesada (2007) where:
    - Local weight loc.w(τ) = log₂(f(τ) + 1) 
    - P_c(τ) = ratio of local frequency to corpus frequency
    - Global weight glob.w = 1 + (sum(P_c(τ) * -log₂(P_c(τ)))) / log₂(|C|)
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
    
    Returns
    -------
    float
        Mean global weight across all m-types, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)

    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    global_weights = []
    
    # Calculate for each n-gram length
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        
        if not ngram_counts:
            continue
            
        # Get total corpus count for normalization
        corpus_total = sum(item['count'] for item in corpus_stats['document_frequencies'] 
                         if item['n'] == n)
        
        if corpus_total == 0:
            continue
            
        # Calculate P_c for each n-gram
        pc_values = []
        for ngram, local_freq in ngram_counts.items():
            df = get_ngram_document_frequency(ngram, corpus_stats)
            
            if df > 0:
                pc = local_freq / df
                pc_values.append((pc, local_freq))
        
        if pc_values:
            # Calculate global weight
            entropy_sum = sum(pc * -np.log2(pc) for pc, _ in pc_values)
            glob_w = 1 + (entropy_sum / np.log2(corpus_total))
            
            # Calculate weighted average using local frequencies
            total_freq = sum(freq for _, freq in pc_values)
            weighted_glob_w = glob_w * total_freq
            global_weights.append(weighted_glob_w)
    
    return float(np.mean(global_weights)) if global_weights else 0.0

def compute_mtcf_std_g_weight(melody: Melody) -> float:
    """Calculate standard deviation of global weights across m-types.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Standard deviation of global weights, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)

    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    global_weights = []
    
    # Calculate for each n-gram length
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        
        if not ngram_counts:
            continue
            
        # Get total corpus count for normalization
        corpus_total = sum(item['count'] for item in corpus_stats['document_frequencies'] 
                         if item['n'] == n)
        
        if corpus_total == 0:
            continue
            
        # Calculate P_c for each n-gram
        for ngram, local_freq in ngram_counts.items():
            df = get_ngram_document_frequency(ngram, corpus_stats)
            
            if df > 0:
                pc = local_freq / df
                # Calculate global weight for this n-gram
                entropy = -np.log2(pc)
                glob_w = 1 + (entropy / np.log2(corpus_total))
                global_weights.append(glob_w)
    
    return float(np.std(global_weights)) if global_weights else 0.0

def compute_mtcf_mean_gl_weight(melody: Melody) -> float:
    """Calculate mean global-local weight across m-types.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Mean global-local weight, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)

    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    gl_weights = []
    
    # Calculate for each n-gram length
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        
        if not ngram_counts:
            continue
            
        # Get total corpus count for normalization
        corpus_total = sum(item['count'] for item in corpus_stats['document_frequencies'] 
                         if item['n'] == n)
        
        if corpus_total == 0:
            continue
            
        # Calculate global-local weight for each n-gram
        for ngram, local_freq in ngram_counts.items():
            df = get_ngram_document_frequency(ngram, corpus_stats)
            
            if df > 0:
                pc = local_freq / df
                # Calculate global weight
                entropy = -np.log2(pc)
                glob_w = 1 + (entropy / np.log2(corpus_total))
                # Multiply by local weight (frequency)
                gl_weight = glob_w * local_freq
                gl_weights.append(gl_weight)
    
    return float(np.mean(gl_weights)) if gl_weights else 0.0

def compute_mtcf_std_gl_weight(melody: Melody) -> float:
    """Calculate standard deviation of global-local weights across m-types.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    float
        Standard deviation of global-local weights, or 0.0 if no n-grams found
    """
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)

    # Load corpus statistics
    with open('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json', encoding='utf-8') as f:
        corpus_stats = json.load(f)

    gl_weights = []
    
    # Calculate for each n-gram length
    for n in range(1, 5):
        ngram_counts = tokenizer.ngram_counts(n=n)
        
        if not ngram_counts:
            continue
            
        # Get total corpus count for normalization
        corpus_total = sum(item['count'] for item in corpus_stats['document_frequencies'] 
                         if item['n'] == n)
        
        if corpus_total == 0:
            continue
            
        # Calculate global-local weight for each n-gram
        for ngram, local_freq in ngram_counts.items():
            df = get_ngram_document_frequency(ngram, corpus_stats)
            
            if df > 0:
                pc = local_freq / df
                # Calculate global weight
                entropy = -np.log2(pc)
                glob_w = 1 + (entropy / np.log2(corpus_total))
                # Multiply by local weight (frequency)
                gl_weight = glob_w * local_freq
                gl_weights.append(gl_weight)
    
    return float(np.std(gl_weights)) if gl_weights else 0.0


def get_corpus_features(melody: Melody) -> Dict:
    """Compute all corpus-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of corpus-based feature values
    """

    return {
        'tfdf_spearman': compute_tfdf_spearman(melody),
        'tfdf_kendall': compute_tfdf_kendall(melody),
        'mean_log_tfdf': compute_tfdf(melody),
        'norm_log_dist': compute_norm_log_dist(melody),
        'max_log_df': compute_max_log_df(melody),
        'min_log_df': compute_min_log_df(melody),
        'mean_log_df': compute_mean_log_df(melody),
        'mean_df_entropy': compute_mean_df_entropy(melody),
        'mean_df_productivity': compute_mean_df_productivity(melody),
        'mean_df_yules_k': compute_mean_df_yules_k(melody),
        'mean_df_simpsons_d': compute_mean_df_simpsons_d(melody),
        'mean_df_sichels_s': compute_mean_df_sichels_s(melody),
        'mean_df_honores_h': compute_mean_df_honores_h(melody)
    }

def get_pitch_features(melody: Melody) -> Dict:
    """Compute all pitch-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of pitch-based feature values
    
    """
    pitch_features = {}
    
    pitch_features['pitch_range'] = pitch_range(melody.pitches)
    pitch_features['pitch_standard_deviation'] = pitch_standard_deviation(melody.pitches)
    pitch_features['pitch_entropy'] = pitch_entropy(melody.pitches)
    pitch_features['pcdist1'] = pcdist1(melody.pitches, melody.starts, melody.ends)
    pitch_features['basic_pitch_histogram'] = basic_pitch_histogram(melody.pitches)
    pitch_features['mean_pitch'] = mean_pitch(melody.pitches)
    pitch_features['most_common_pitch'] = most_common_pitch(melody.pitches)
    pitch_features['number_of_pitches'] = number_of_pitches(melody.pitches)
    pitch_features['melodic_pitch_variety'] = melodic_pitch_variety(melody.pitches)
    pitch_features['dominant_spread'] = dominant_spread(melody.pitches)
    pitch_features['folded_fifths_pitch_class_histogram'] = folded_fifths_pitch_class_histogram(melody.pitches)
    pitch_features['pitch_class_kurtosis_after_folding'] = pitch_class_kurtosis_after_folding(melody.pitches)
    pitch_features['pitch_class_skewness_after_folding'] = pitch_class_skewness_after_folding(melody.pitches)
    pitch_features['pitch_class_variability_after_folding'] = pitch_class_variability_after_folding(melody.pitches)
    
    return pitch_features

def get_interval_features(melody: Melody) -> Dict:
    """Compute all interval-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of interval-based feature values
    
    """
    interval_features = {}
    
    interval_features['pitch_interval'] = pitch_interval(melody.pitches)
    interval_features['absolute_interval_range'] = absolute_interval_range(melody.pitches)
    interval_features['mean_absolute_interval'] = mean_absolute_interval(melody.pitches)
    interval_features['modal_interval'] = modal_interval(melody.pitches)
    interval_features['interval_entropy'] = interval_entropy(melody.pitches)
    interval_features['ivdist1'] = ivdist1(melody.pitches, melody.starts, melody.ends)
    direction_mean, direction_sd = interval_direction(melody.pitches)
    interval_features['interval_direction_mean'] = direction_mean
    interval_features['interval_direction_sd'] = direction_sd
    interval_features['average_interval_span_by_melodic_arcs'] = average_interval_span_by_melodic_arcs(melody.pitches)
    interval_features['distance_between_most_prevalent_melodic_intervals'] = distance_between_most_prevalent_melodic_intervals(melody.pitches)
    interval_features['melodic_interval_histogram'] = melodic_interval_histogram(melody.pitches)
    interval_features['melodic_large_intervals'] = melodic_large_intervals(melody.pitches)
    interval_features['variable_melodic_intervals'] = variable_melodic_intervals(melody.pitches, 13)  # TODO: Add more interval levels
    interval_features['number_of_common_melodic_intervals'] = number_of_common_melodic_intervals(melody.pitches)
    interval_features['prevalence_of_most_common_melodic_interval'] = prevalence_of_most_common_melodic_interval(melody.pitches)
    
    return interval_features

def get_contour_features(melody: Melody) -> Dict:
    """Compute all contour-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of contour-based feature values
    
    """
    contour_features = {}
    
    # Calculate step contour features
    step_contour = get_step_contour_features(melody.pitches, melody.starts, melody.ends)
    contour_features['step_contour_global_variation'] = step_contour[0]
    contour_features['step_contour_global_direction'] = step_contour[1]
    contour_features['step_contour_local_variation'] = step_contour[2]
    
    # Calculate interpolation contour features
    interpolation_contour = get_interpolation_contour_features(melody.pitches, melody.starts)
    contour_features['interpolation_contour_global_direction'] = interpolation_contour[0]
    contour_features['interpolation_contour_mean_gradient'] = interpolation_contour[1]
    contour_features['interpolation_contour_gradient_std'] = interpolation_contour[2]
    contour_features['interpolation_contour_direction_changes'] = interpolation_contour[3]
    contour_features['interpolation_contour_class_label'] = interpolation_contour[4]
    
    return contour_features

def get_duration_features(melody: Melody) -> Dict:
    """Compute all duration-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of duration-based feature values
    
    """
    duration_features = {}
    duration_features['tempo'] = get_tempo(melody)
    duration_features['duration_range'] = duration_range(melody.starts, melody.ends)
    duration_features['modal_duration'] = modal_duration(melody.starts, melody.ends)
    duration_features['mean_duration'] = mean_duration(melody.starts, melody.ends)
    duration_features['duration_standard_deviation'] = duration_standard_deviation(melody.starts, melody.ends)
    duration_features['number_of_durations'] = number_of_durations(melody.starts, melody.ends)
    duration_features['global_duration'] = global_duration(melody.starts, melody.ends)
    duration_features['note_density'] = note_density(melody.starts, melody.ends)
    duration_features['duration_entropy'] = duration_entropy(melody.starts, melody.ends)
    duration_features['length'] = length(melody.starts)
    duration_features['note_density'] = note_density(melody.starts, melody.ends)
    duration_features['ioi_mean'] = ioi_mean(melody.starts)
    duration_features['ioi_std'] = ioi_standard_deviation(melody.starts)
    ioi_ratio_mean, ioi_ratio_std = ioi_ratio(melody.starts)
    duration_features['ioi_ratio_mean'] = ioi_ratio_mean
    duration_features['ioi_ratio_std'] = ioi_ratio_std
    duration_features['ioi_contour'] = ioi_contour(melody.starts)
    duration_features['ioi_range'] = ioi_range(melody.starts)
    duration_features['ioi_histogram'] = ioi_histogram(melody.starts)
    duration_features['duration_histogram'] = duration_histogram(melody.starts, melody.ends)
    return duration_features

def get_tonality_features(melody: Melody) -> Dict:
    """Compute all tonality-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of tonality-based feature values
    
    """
    tonality_features = {}
    
    tonality_features['tonalness'] = tonalness(melody.pitches)
    tonality_features['tonal_clarity'] = tonal_clarity(melody.pitches)
    tonality_features['tonal_spike'] = tonal_spike(melody.pitches)
    tonality_features['tonal_entropy'] = tonal_entropy(melody.pitches)
    tonality_features['referent'] = referent(melody.pitches)
    tonality_features['inscale'] = inscale(melody.pitches)
    tonality_features['temperley_likelihood'] = temperley_likelihood(melody.pitches)
    tonality_features['longest_monotonic_conjunct_scalar_passage'] = longest_monotonic_conjunct_scalar_passage(melody.pitches)
    tonality_features['longest_conjunct_scalar_passage'] = longest_conjunct_scalar_passage(melody.pitches)
    tonality_features['proportion_conjunct_scalar'] = proportion_conjunct_scalar(melody.pitches)
    tonality_features['proportion_scalar'] = proportion_scalar(melody.pitches)
    tonality_features['tonalness_histogram'] = tonalness_histogram(melody.pitches)
    return tonality_features

def get_melodic_movement_features(melody: Melody) -> Dict:
    """Compute all melodic movement-based features for a melody.
    
    Parameters
    ----------
    melody : Melody
        The melody to analyze
        
    Returns
    -------
    Dict
        Dictionary of melodic movement-based feature values
    
    """
    movement_features = {}
    
    movement_features['amount_of_arpeggiation'] = amount_of_arpeggiation(melody.pitches)
    movement_features['chromatic_motion'] = chromatic_motion(melody.pitches)
    movement_features['melodic_embellishment'] = melodic_embellishment(melody.pitches, melody.starts, melody.ends)
    movement_features['repeated_notes'] = repeated_notes(melody.pitches)
    movement_features['stepwise_motion'] = stepwise_motion(melody.pitches)
    
    return movement_features

def get_all_features_json(filename) -> Dict:
    with open("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/mididata5.json", encoding='utf-8') as f:
        melody_data_list = json.load(f)
        print(f"Processing {len(melody_data_list)} melodies")
    
    features_by_melody = {}
    for i, melody_data in enumerate(melody_data_list, 1):
        mel = Melody(melody_data, tempo=100)
        melody_features = {
            'pitch_features': get_pitch_features(mel),
            'interval_features': get_interval_features(mel),
            'contour_features': get_contour_features(mel),
            'duration_features': get_duration_features(mel), 
            'tonality_features': get_tonality_features(mel),
            'narmour_features': get_narmour_features(mel),
            'melodic_movement_features': get_melodic_movement_features(mel),
            'mtype_features': get_mtype_features(mel),
            'corpus_features': get_corpus_features(mel)
        }
        features_by_melody[f'id: {i}'] = melody_features
        print(f"Processed melody {i}/{len(melody_data_list)}")

    output_file = f'{filename.rsplit(".", 1)[0]}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(features_by_melody, f, indent=2)
    print(f"Features saved to {output_file}")

def process_melody(args):
    """Process a single melody and return its features.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (melody_id, melody_data)
    
    Returns
    -------
    tuple
        Tuple containing (melody_id, feature_dict)
    """
    melody_id, melody_data = args
    mel = Melody(melody_data, tempo=100)
    
    melody_features = {
        'pitch_features': get_pitch_features(mel),
        'interval_features': get_interval_features(mel),
        'contour_features': get_contour_features(mel),
        'duration_features': get_duration_features(mel),
        'tonality_features': get_tonality_features(mel),
        'narmour_features': get_narmour_features(mel),
        'melodic_movement_features': get_melodic_movement_features(mel),
        'mtype_features': get_mtype_features(mel),
        'corpus_features': get_corpus_features(mel)
    }
    
    return melody_id, melody_features

def get_all_features_csv(input_path, output_path) -> None:
    """Generate CSV file with features for all melodies using multiprocessing.
    
    Parameters
    ----------
    input_path : str
        Path to input JSON file
    output_path : str
        Path to output CSV file
        
    Returns
    -------
    None
        Writes features to CSV file with each melody as a row
    """
    print("Starting job...\n")
    with open(input_path, encoding='utf-8') as f:
        melody_data_list = json.load(f)
        print(f"Processing {len(melody_data_list)} melodies")

    start_time = time.time()

    # Process first melody to get header structure
    mel = Melody(melody_data_list[0], tempo=100)
    first_features = {
        'pitch_features': get_pitch_features(mel),
        'interval_features': get_interval_features(mel),
        'contour_features': get_contour_features(mel),
        'duration_features': get_duration_features(mel),
        'tonality_features': get_tonality_features(mel),
        'narmour_features': get_narmour_features(mel),
        'melodic_movement_features': get_melodic_movement_features(mel),
        'mtype_features': get_mtype_features(mel),
        'corpus_features': get_corpus_features(mel)
    }

    # Create header by flattening feature names
    headers = ['melody_id']
    for category, features in first_features.items():
        headers.extend(f"{category}.{feature}" for feature in features.keys())

    print("Starting parallel processing...\n")
    # Create pool of workers
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores")

    # Prepare arguments for parallel processing
    melody_args = list(enumerate(melody_data_list, 1))
    
    # Process melodies in parallel
    all_features = []
    
    with Pool(n_cores) as pool:
        # Use imap_unordered for better performance and wrap with tqdm
        for melody_id, melody_features in tqdm(
            pool.imap_unordered(process_melody, melody_args),
            total=len(melody_args),
            desc="Processing melodies",
            mininterval=0.2
        ):
            # Flatten feature values into a single row
            row = [melody_id]
            for category, features in melody_features.items():
                row.extend(features.values())
            all_features.append(row)

    # Sort results by melody_id since they completed in different order
    all_features.sort(key=lambda x: x[0])
    
    # Write to CSV
    output_file = f'{output_path}.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_features)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(melody_data_list)
    
    print(f"\nFeatures saved to {output_file}")
    print(f"Generated in total time: {total_time:.2f} seconds")
    print(f"Average time per melody: {avg_time:.3f} seconds\n")

    print("Job complete\n")

if __name__ == "__main__":
    # get_all_features_json('item_features')
    get_all_features_csv(
        input_path='/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/mididata5.json',
        output_path='item_features2')
