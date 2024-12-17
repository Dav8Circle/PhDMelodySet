"""
Implements features derived from Narmour (1990) concerning Gestalt principles for melody.
"""
__author__ = "David Whyatt"

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
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

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
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

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
        return 3 - abs(pitch3 - pitch1)

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
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

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
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

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
        return (1 if abs(realized) >= abs(implicative) - margin and
                abs(realized) <= abs(implicative) + margin else 0)
    else:
        return 0


def closure(values: list[float]) -> float:
    """Calculates the closure score based on the shape defined by the last three notes.
    
    Scores 1 point for a change of direction and 1 point for an interval that is more than 
    a tone smaller than the preceding one.
    
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
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

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
