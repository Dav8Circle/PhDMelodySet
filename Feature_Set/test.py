import pytest
from algorithms import *


def test_range():
    assert range_func([1, 2, 3]) == 2
    assert range_func([1, 1, 1]) == 0
    assert range_func([]) == 0
    assert range_func([1, 2, 3, 4, 5]) == 4
    with pytest.raises(TypeError):
        range_func([1, 2, "a", "b"])

def test_standard_deviation():
    assert standard_deviation([1, 2, 3]) == pytest.approx(0.816496580927726)
    assert standard_deviation([1, 1, 1]) == 0
    assert standard_deviation([]) == 0
    assert standard_deviation([1, 2, 3, 4, 5]) == pytest.approx(1.4142135623730951) 
    with pytest.raises(TypeError):
        standard_deviation([1, 2, "a", "b"])


def test_shannon_entropy():
    assert shannon_entropy([1, 2, 3]) == pytest.approx(1.5849625007211563)
    assert shannon_entropy([1, 1, 1]) == 0  # All same values = 0 entropy
    assert shannon_entropy([]) == 0  # Empty list = 0 entropy
    assert shannon_entropy([1, 2, 2, 3, 3, 3]) == pytest.approx(1.4591479170272448)
    with pytest.raises(TypeError):
        shannon_entropy([1, 2, "a", "b"])

def test_natural_entropy():
    assert natural_entropy([1, 2, 3]) == pytest.approx(1.0986122886681098)  # ln-based entropy
    assert natural_entropy([1, 1, 1]) == 0  # All same values = 0 entropy
    assert natural_entropy([]) == 0  # Empty list = 0 entropy
    assert natural_entropy([1, 2, 2, 3, 3, 3]) == pytest.approx(1.0114042647073518)
    with pytest.raises(TypeError):
        natural_entropy([1, 2, "a", "b"])

def test_distribution_proportions():
    assert distribution_proportions([1, 2, 3]) == {1.0: 0.3333333333333333, 2.0: 0.3333333333333333, 3.0: 0.3333333333333333}
    assert distribution_proportions([1, 1, 1]) == {1.0: 1.0}  # Single value has 100% proportion
    assert distribution_proportions([]) == {}  # Empty list returns empty dict
    assert distribution_proportions([1, 2, 2, 3, 3, 3]) == {1.0: 0.16666666666666666, 2.0: 0.3333333333333333, 3.0: 0.5}
    with pytest.raises(TypeError):
        distribution_proportions([1, 2, "a", "b"])

def test_ratio():
    # Basic test
    assert ratio([2, 4, 6], [1, 2, 3]) == [2.0, 2.0, 2.0]
    
    # Empty lists return empty list
    assert ratio([], []) == []
    
    # Different length lists return empty list
    assert ratio([1, 2], [1, 2, 3]) == []
    
    # Division by zero returns None
    assert ratio([1, 2, 3], [1, 0, 2]) == [1.0, None, 1.5]
    
    # Handles floats
    assert ratio([1.5, 3.0, 4.5], [0.5, 1.0, 1.5]) == [3.0, 3.0, 3.0]
    
    # Handles negative numbers
    assert ratio([-2, -4, -6], [1, 2, 3]) == [-2.0, -2.0, -2.0]
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        ratio([1, 2, "a"], [1, 2, 3])
    with pytest.raises(TypeError):
        ratio([1, 2, 3], [1, "b", 3])


def test_rank_values():
    # Test ascending ranks (default)
    assert rank_values([1, 2, 3]) == [1, 2, 3]
    assert rank_values([3, 1, 2]) == [3, 1, 2]
    assert rank_values([1, 1, 1]) == [1, 1, 1]  # Ties get same rank
    assert rank_values([]) == []  # Empty list returns empty list
    assert rank_values([1, 2, 2, 3]) == [1, 2, 2, 3]  # Handles ties correctly
    
    # Test descending ranks
    assert rank_values([1, 2, 3], descending=True) == [3, 2, 1]
    assert rank_values([3, 1, 2], descending=True) == [1, 3, 2]
    assert rank_values([1, 2, 2, 3], descending=True) == [3, 2, 2, 1]
    
    # Test error cases
    with pytest.raises(TypeError):
        rank_values([1, 2, "a", "b"])

def test_repetition_rate():
    assert repetition_rate([1, 2, 3, 1]) == 3  # First 1 to second 1 is distance 3
    assert repetition_rate([1, 1, 1]) == 1  # Adjacent repeats have distance 1
    assert repetition_rate([1, 2, 3]) == 0  # No repeats returns 0
    assert repetition_rate([]) == 0  # Empty list returns 0
    assert repetition_rate([1, 2, 1, 2]) == 2  # Multiple repeats, returns shortest
    assert repetition_rate([1, 2, 3, 2, 1]) == 2  # Returns first occurrence of shortest
    with pytest.raises(TypeError):
        repetition_rate([1, 2, "a", "b"])

def test_repetition_count():
    assert repetition_count([1, 2, 3, 1]) == {1.0: 2}  # One number (1) appears twice
    assert repetition_count([1, 1, 1]) == {1.0: 3}  # One number appears three times
    assert repetition_count([1, 2, 3]) == {}  # No repetitions
    assert repetition_count([]) == {}  # Empty list returns empty dict
    assert repetition_count([1, 2, 1, 2]) == {1.0: 2, 2.0: 2}  # Both numbers appear twice
    assert repetition_count([1, 2, 2, 2, 1]) == {1.0: 2, 2.0: 3}  # One appears twice, other three times
    with pytest.raises(TypeError):
        repetition_count([1, 2, "a", "b"])

def test_consecutive_repetition_count():
    assert consecutive_repetition_count([1, 1, 2, 2, 2, 1]) == {1.0: 2, 2.0: 3}  # Two 1s then three 2s
    assert consecutive_repetition_count([1, 2, 3]) == {}  # No consecutive repeats
    assert consecutive_repetition_count([]) == {}  # Empty list returns empty dict
    assert consecutive_repetition_count([1, 1, 1]) == {1.0: 3}  # Three consecutive 1s
    assert consecutive_repetition_count([1, 1, 2, 3, 3]) == {1.0: 2, 3.0: 2}  # Two pairs of repeats
    assert consecutive_repetition_count([1, 2, 2, 1, 1]) == {2.0: 2, 1.0: 2}  # Non-consecutive 1s don't combine
    with pytest.raises(TypeError):
        consecutive_repetition_count([1, 2, "a", "b"])

def test_mean():
    assert mean([1, 2, 3]) == 2.0  # Basic mean calculation
    assert mean([1, 1, 1]) == 1.0  # Same numbers
    assert mean([]) == 0  # Empty list returns 0
    assert mean([1.5, 2.5]) == 2.0  # Handles floats
    assert mean([-1, 0, 1]) == 0.0  # Handles negative numbers
    with pytest.raises(TypeError):
        mean([1, 2, "a", "b"])

def test_mode():
    assert mode([1, 2, 2, 3]) == 2.0  # Basic mode calculation
    assert mode([1, 1, 2, 2]) == 1.0  # Multiple modes returns smallest
    assert mode([]) == 0  # Empty list returns 0
    assert mode([1.5, 1.5, 2.5]) == 1.5  # Handles floats
    assert mode([-1, -1, 0, 1]) == -1.0  # Handles negative numbers
    assert mode([1, 1, 1]) == 1.0  # All same number
    with pytest.raises(TypeError):
        mode([1, 2, "a", "b"])

def test_length():
    assert length([1, 2, 3]) == 3  # Basic length calculation
    assert length([]) == 0  # Empty list returns 0
    assert length([1.5, 2.5]) == 2  # Handles floats
    with pytest.raises(TypeError):
        length([1, 2, "a", "b"])


def test_modulo_twelve():
    assert modulo_twelve([12, 13, 24, 25]) == [0.0, 1.0, 0.0, 1.0]  # Basic modulo calculation
    assert modulo_twelve([]) == []  # Empty list returns empty list
    assert modulo_twelve([11.5, 12.5]) == [11.5, 0.5]  # Handles floats
    with pytest.raises(TypeError):
        modulo_twelve([1, 2, "a", "b"])

def test_histogram_bins():
    # Basic histogram with 2 bins
    assert histogram_bins([1, 1.5, 2, 2.5, 3], 2) == {'1.00-2.00': 2, '2.00-3.00': 3}
    
    # Empty list returns empty dict
    assert histogram_bins([], 5) == {}
    
    # Single bin contains all values
    assert histogram_bins([1, 2, 3], 1) == {'1.00-3.00': 3}
    
    # Handles negative numbers
    assert histogram_bins([-2, -1, 0, 1, 2], 2) == {'-2.00-0.00': 2, '0.00-2.00': 3}
    
    # Handles floats
    assert histogram_bins([0.5, 1.5, 2.5], 3) == {'0.50-1.17': 1, '1.17-1.83': 1, '1.83-2.50': 1}
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        histogram_bins([1, 2, "a", "b"], 2)
    with pytest.raises(ValueError):
        histogram_bins([1, 2, 3], 0)

def test_standardize_distribution():
    # Basic standardization - should have mean 0 and std dev 1
    result = standardize_distribution([1, 2, 3, 4, 5])
    assert np.allclose(np.mean(result), 0)
    assert np.allclose(np.std(result), 1)
    
    # Empty list returns empty list
    assert standardize_distribution([]) == []
    
    # Handles negative numbers
    result = standardize_distribution([-2, -1, 0, 1, 2])
    assert np.allclose(np.mean(result), 0)
    assert np.allclose(np.std(result), 1)
    
    # Handles floats
    result = standardize_distribution([1.5, 2.5, 3.5])
    assert np.allclose(np.mean(result), 0)
    assert np.allclose(np.std(result), 1)
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        standardize_distribution([1, 2, "a", "b"])
        
    # Raises error when std dev is 0 (all same numbers)
    with pytest.raises(ValueError):
        standardize_distribution([2, 2, 2])

def test_normalize_distribution():
    # Basic normalization - should be between 0 and 1
    result, mean, std = normalize_distribution([1, 2, 3, 4, 5])
    assert all(0 <= x <= 1 for x in result)
    assert np.allclose(result, [0, 0.25, 0.5, 0.75, 1])
    assert np.allclose(mean, 0.5)
    assert np.allclose(std, np.std([0, 0.25, 0.5, 0.75, 1]))
    
    # Empty list returns empty list and zeros
    assert normalize_distribution([]) == ([], 0, 0)
    
    # Handles negative numbers
    result, mean, std = normalize_distribution([-2, -1, 0, 1, 2])
    assert all(0 <= x <= 1 for x in result)
    assert np.allclose(result, [0, 0.25, 0.5, 0.75, 1])
    
    # Handles floats
    result, mean, std = normalize_distribution([1.5, 2.5, 3.5])
    assert all(0 <= x <= 1 for x in result)
    assert np.allclose(result, [0, 0.5, 1])
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        normalize_distribution([1, 2, "a", "b"])
        
    # Raises error when all values are identical
    with pytest.raises(ValueError):
        normalize_distribution([2, 2, 2])

def test_kurtosis():
    # Normal distribution has kurtosis close to 0
    assert np.allclose(kurtosis([-3, -2, -1, 0, 0, 1, 1, 2, 3]), 0, atol=0.5)
    
    # Empty list returns 0
    assert kurtosis([]) == 0
    
    # List with less than 2 unique values returns 0 
    assert kurtosis([2, 2, 2]) == 0
    assert kurtosis([]) == 0
    
    # Handles negative numbers
    assert isinstance(kurtosis([-2, -1, 0, 1, 2]), float)
    
    # Handles floats
    assert isinstance(kurtosis([1.5, 2.5, 3.5]), float)
    
    # High kurtosis (peaked distribution with heavy tails)
    high_kurt = [1, 1, 1, 5, 1, 1, 1]
    assert kurtosis(high_kurt) > 0

    # Low kurtosis (uniform distribution)
    low_kurt = [1, 1.5, 2, 2.5, 3]
    assert kurtosis(low_kurt) < 0
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        kurtosis([1, 2, "a", "b"])

def test_skew():
    # Normal distribution has skew close to 0
    assert np.allclose(skew([-3, -2, -1, 0, 0, 1, 1, 2, 3]), 0, atol=0.5)
    
    # Empty list returns 0
    assert skew([]) == 0
    
    # List with less than 2 unique values returns 0
    assert skew([2, 2, 2]) == 0
    assert skew([]) == 0
    
    # Handles negative numbers
    assert isinstance(skew([-2, -1, 0, 1, 2]), float)
    
    # Handles floats 
    assert isinstance(skew([1.5, 2.5, 3.5]), float)
    
    # Positive skew (longer tail on right)
    right_skewed = [1, 1, 1, 1, 5]
    assert skew(right_skewed) > 0
    
    # Negative skew (longer tail on left)
    left_skewed = [0, 4, 4, 4, 4]
    assert skew(left_skewed) < 0
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        skew([1, 2, "a", "b"])

def test_absolute_values():
    # Basic test with positive and negative numbers
    assert absolute_values([1, -2, 3, -4]) == [1, 2, 3, 4]
    
    # Empty list returns empty list
    assert absolute_values([]) == []
    
    # All positive numbers remain unchanged
    assert absolute_values([1, 2, 3]) == [1, 2, 3]
    
    # All negative numbers become positive
    assert absolute_values([-1, -2, -3]) == [1, 2, 3]
    
    # Handles zero
    assert absolute_values([-1, 0, 1]) == [1, 0, 1]
    
    # Handles floats
    assert absolute_values([-1.5, 2.5, -3.5]) == [1.5, 2.5, 3.5]
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        absolute_values([1, 2, "a", "b"])

def test_limit():
    # Basic test with both limits
    assert limit([1, 2, 3, 4, 5], x=4, y=2) == [2, 3, 4]
    
    # Empty list returns empty list
    assert limit([]) == []
    
    # Only upper limit
    assert limit([1, 2, 3, 4, 5], x=3) == [1, 2, 3]
    
    # Only lower limit 
    assert limit([1, 2, 3, 4, 5], y=3) == [3, 4, 5]
    
    # No limits returns original list
    assert limit([1, 2, 3]) == [1, 2, 3]
    
    # Handles negative numbers
    assert limit([-3, -2, -1, 0, 1, 2, 3], x=1, y=-2) == [-2, -1, 0, 1]
    
    # Handles floats
    assert limit([1.5, 2.5, 3.5, 4.5], x=4.0, y=2.0) == [2.5, 3.5]
    
    # Inclusive limits
    assert limit([1, 2, 3, 4, 5], x=5, y=1) == [1, 2, 3, 4, 5]
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        limit([1, 2, "a", "b"])


def test_correlation():
    # Basic positive correlation
    assert correlation([1, 2, 3], [2, 4, 6]) == 1.0
    
    # Basic negative correlation
    assert correlation([1, 2, 3], [-2, -4, -6]) == -1.0

    # No correlation (zero variance in second list)
    assert correlation([1, 2, 3], [1, 1, 1]) == 0.0
    
    # Empty lists return None
    assert correlation([], []) is None
    
    # Different length lists return None
    assert correlation([1, 2], [1, 2, 3]) is None
    
    # Handles floats
    assert abs(correlation([1.5, 2.5, 3.5], [1.0, 2.0, 3.0]) - 1.0) < 0.0001
    
    # Handles negative numbers
    assert abs(correlation([-1, -2, -3], [-2, -4, -6]) - 1.0) < 0.0001
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        correlation([1, 2, "a"], [1, 2, 3])
    with pytest.raises(TypeError):
        correlation([1, 2, 3], [1, "b", 3])

def test_nine_percent_significant_values():
    # Basic test - value appears exactly 9% of time (1/11)
    assert nine_percent_significant_values([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]) == [2]
    
    # Multiple values above 9% threshold
    assert nine_percent_significant_values([1, 1, 1, 2, 2, 2, 2, 2, 3, 3]) == [1, 2, 3]
    
    # No values above 9% threshold (each appears 8.3%)
    assert nine_percent_significant_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) == []
    
    # Empty list returns empty list
    assert nine_percent_significant_values([]) == []
    
    # Custom threshold overriding default 9%
    assert nine_percent_significant_values([1, 2, 2, 3, 3, 3], threshold=0.45) == [3]

    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        nine_percent_significant_values([1, 2, "a", "b"])

def test_contour_extrema():
    # Basic test with clear maxima and minima
    assert contour_extrema([1, 3, 2, 4, 1, 5, 2]) == [(0, 1), (1, 3), (2, 2), (3, 4), (4, 1), (5, 5), (6, 2)]
    
    # Test with plateau
    assert contour_extrema([1, 2, 2, 2, 1]) == [(0, 1), (4, 1)]
    
    # Empty list returns empty list
    assert contour_extrema([]) == []
    
    # Single value returns empty list
    assert contour_extrema([1]) == []
    
    # Two values returns both points
    assert contour_extrema([1, 2]) == [(0, 1), (1, 2)]
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        contour_extrema([1, "a", 2])

def test_contour_gradients():
    # Basic test with clear slopes
    assert contour_gradients([1, 3, 2, 4, 1, 5, 2]) == [2.0, -1.0, 2.0, -3.0, 4.0, -3.0]
    
    # Test with plateau
    gradients = contour_gradients([1, 2, 2, 2, 1])
    assert len(gradients) == 1
    
    # Empty list returns empty list
    assert contour_gradients([]) == []
    
    # Single value returns empty list
    assert contour_gradients([1]) == []
    
    # Two values returns single gradient
    assert contour_gradients([1, 2]) == [1.0]
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        contour_gradients([1, "a", 2])

def test_compute_tonality_vector():
    # Empty list returns zeros
    assert compute_tonality_vector([]) == [0] * 24
    
    # Test with C major scale (pitch classes 0,2,4,5,7,9,11)
    c_major = [0, 2, 4, 5, 7, 9, 11]
    result = compute_tonality_vector(c_major)
    # C major (index 0) should have highest correlation
    assert result[0] == max(result)
    
    # Test with A minor scale (pitch classes 9,11,0,2,4,5,7)
    a_minor = [9, 11, 0, 2, 4, 5, 7] 
    result = compute_tonality_vector(a_minor)
    # A minor (index 21) should have high correlation
    assert result[21] in sorted(result)[-3:]
    
    # Test with chromatic scale
    chromatic = list(range(12))
    result = compute_tonality_vector(chromatic)
    # All correlations should be similar due to equal distribution
    assert max(result) - min(result) < 0.5
    
    # Raises error for invalid inputs
    with pytest.raises(TypeError):
        compute_tonality_vector([0, 1, 13])  # Invalid pitch class
    with pytest.raises(TypeError):
        compute_tonality_vector([0, 1, "C"])  # Non-integer


if __name__ == '__main__':
    pytest.main([__file__])

