import numpy as np
from scipy import stats, signal

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
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Check for zero variance in either array
    if np.var(x_array) == 0 or np.var(y_array) == 0:
        return 0.0

    # Use numpy's built-in corrcoef function which implements Pearson correlation
    correlation_matrix = np.corrcoef(x_array, y_array)

    # corrcoef returns a 2x2 matrix, we want the off-diagonal element
    return float(correlation_matrix[0, 1])


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
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    corr, _ = stats.spearmanr(x_array, y_array)

    # Handle NaN result
    if np.isnan(corr):
        return 0.0

    return corr

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
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    tau, _ = stats.kendalltau(x_array, y_array)

    # Handle NaN result
    if np.isnan(tau):
        return 0.0

    return tau


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
    except (TypeError, ValueError) as exc:
        raise TypeError("All elements must be numbers") from exc

    # Calculate cross-correlation using scipy.signal.correlate
    corr = signal.correlate(x_array, y_array, mode='full')

    return corr.tolist()