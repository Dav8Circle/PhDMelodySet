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



if __name__ == '__main__':
    pytest.main([__file__])

