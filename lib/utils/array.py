import numpy as np


def safe_normalize(x, eps=1e-20):
    length = np.sqrt(np.clip(np.sum(x*x, axis=-1, keepdims=True), a_min=eps, a_max=None))
    return x / length