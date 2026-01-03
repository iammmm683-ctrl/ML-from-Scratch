import numpy as np

def build_precision_recall_curve(y, p):
    y = np.asarray(y, int)
    if y.sum() == 0:
        raise ValueError()
    p = np.asarray(p, float)
    y = y[np.argsort(-p, kind="mergesort")]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    r = tp / y.sum()
    pr = tp / (tp + fp)
    return np.vstack(([0.0, 1.0], np.column_stack((r, pr))))
