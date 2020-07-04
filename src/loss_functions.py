import torch
import torch.nn as nn


def lossFunction(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))
