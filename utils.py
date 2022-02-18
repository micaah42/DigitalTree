import numpy as np


def logistic(x: np.ndarray, low: float = -1., high: float = 1.) -> float:
    if x < low:
        return 0
    elif x < high:
        return (x - low) / (high - low)
    else:
        return 1
