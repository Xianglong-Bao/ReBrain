import warnings

import numpy as np
from scipy.stats import kurtosis


def morphology_features(windows: np.ndarray) -> np.ndarray:
    x = windows.astype(np.float64)
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var0 = x.var(axis=-1) + 1e-12
    var1 = dx.var(axis=-1) + 1e-12
    var2 = ddx.var(axis=-1) + 1e-12

    line_length = np.abs(dx).sum(axis=-1)
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kurt = kurtosis(x, axis=-1, fisher=True, bias=False)
    kurt = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)
    xc = x - x.mean(axis=-1, keepdims=True)
    zcr = (np.abs(np.diff(np.sign(xc), axis=-1)) > 0).mean(axis=-1)

    feats = np.stack([
        np.log10(line_length + 1e-12),
        np.log10(var0),
        mobility,
        complexity,
        kurt,
        zcr,
    ], axis=-1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
