import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(y_pred, y_true):
    y_pred = np.round(y_pred.cpu().numpy().reshape(-1)).astype(np.uint8)
    y_true = np.round(y_true.cpu().numpy().reshape(-1)).astype(np.uint8)

    zero_indices = list(np.where(y_true == 0)[0])
    y_pred = np.delete(y_pred, zero_indices)
    y_true = np.delete(y_true, zero_indices)
    return accuracy_score(y_pred, y_true)
