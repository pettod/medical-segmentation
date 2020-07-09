import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(y_pred, y_true):
    y_pred = y_pred.cpu().numpy().reshape(-1).astype(np.uint8)
    y_true = y_true.cpu().numpy().reshape(-1).astype(np.uint8)
    return accuracy_score(y_pred, y_true)
