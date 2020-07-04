import glob
from inspect import getmembers, isfunction
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from math import ceil

# Project files
import src.metrics as metrics


def getMetrics():
    metrics_name_and_function_pointers = [
        metric for metric in getmembers(metrics, isfunction)
        if metric[1].__module__ == metrics.__name__]
    return metrics_name_and_function_pointers


def computeMetrics(y_pred, y_true):
    metric_functions = getMetrics()
    metric_scores = {}
    for metric_name, metric_function_pointer in metric_functions:
        metric_scores[metric_name] = metric_function_pointer(y_pred, y_true)
    return metric_scores


def initializeEpochMetrics(epoch):
    metric_functions = getMetrics()
    epoch_metrics = {}
    epoch_metrics["epoch"] = epoch
    epoch_metrics["train_loss"] = 0
    epoch_metrics["valid_loss"] = 0
    for metric_name, _ in metric_functions:
        epoch_metrics[f"train_{metric_name}"] = 0
        epoch_metrics[f"valid_{metric_name}"] = 0
    return epoch_metrics


def updateEpochMetrics(
        y_pred, y_true, loss, epoch_iteration_index, epoch_metrics, mode,
        optimizer=None):
    metric_scores = computeMetrics(y_pred, y_true)
    metric_scores["loss"] = loss
    for key, value in metric_scores.items():
        if type(value) == torch.Tensor:
            value = value.item()

        epoch_metrics[f"{mode}_{key}"] += ((
            value - epoch_metrics[f"{mode}_{key}"]) /
            (epoch_iteration_index + 1))
    if optimizer:
        epoch_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]


def getProgressbarText(epoch_metrics, mode):
    text = f" {mode}:"
    mode = mode.lower()
    for key, value in epoch_metrics.items():
        if mode not in key:
            continue
        text += " {}: {:2.4f}.".format(key.replace(f"{mode}_", ""), value)
    return text


def saveLearningCurve(
        log_file_path=None, model_root="saved_models", xticks_limit=13):
    # Read CSV log file
    if log_file_path is None:
        log_file_path = sorted(glob.glob(os.path.join(
            model_root, *['*', "*.csv"])))[-1]
    log_file = pd.read_csv(log_file_path)

    # Read data into dictionary
    log_data = {}
    for column in log_file:
        if column == "epoch":
            log_data[column] = np.array(log_file[column].values, dtype=np.str)
        elif column == "learning_rate":
            continue
        else:
            log_data[column] = np.array(log_file[column].values)
    number_of_epochs = log_file.shape[0]

    # Remove extra printings of same epoch
    used_xticks = [i for i in range(number_of_epochs)]
    epoch_string_data = []
    previous_epoch = -1
    for i, epoch in enumerate(reversed(log_data["epoch"])):
        if epoch != previous_epoch:
            epoch_string_data.append(epoch)
        else:
            used_xticks.pop(-1*i - 1)
        previous_epoch = epoch
    epoch_string_data = epoch_string_data[::-1]
    log_data.pop("epoch", None)

    # Limit number of printed epochs in x axis
    used_xticks = used_xticks[::ceil(number_of_epochs / xticks_limit)]
    epoch_string_data = epoch_string_data[::ceil(
        number_of_epochs / xticks_limit)]

    # Define train and validation subplots
    figure_dict = {}
    for key in log_data.keys():
        metric = key.split('_')[-1]
        if metric not in figure_dict:
            figure_dict[metric] = len(figure_dict.keys()) + 1
    number_of_subplots = len(figure_dict.keys())

    # Save learning curves plot
    plt.figure(figsize=(15, 7))
    import warnings
    warnings.filterwarnings("ignore")
    for i, key in enumerate(log_data.keys()):
        metric = key.split('_')[-1]
        plt.subplot(1, number_of_subplots, figure_dict[metric])
        plt.plot(range(number_of_epochs), log_data[key], label=key)
        plt.xticks(used_xticks, epoch_string_data)
        plt.xlabel("Epoch")
        plt.title(metric.upper())
        plt.legend()
    plt.tight_layout()
    plt.savefig("{}.{}".format(log_file_path.split('.')[0], "png"))


def loadModel(
        model, model_root, model_path=None, optimizer=None,
        load_pretrained_weights=True):
    print("{:,} model parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if not load_pretrained_weights:
        return
    # Load latest model
    if model_path is None:
        model_name = sorted(glob.glob(os.path.join(
            model_root, *['*', "*.pt"])))[-1]
    else:

        # Load model based on index
        if type(model_path) == int:
            model_name = sorted(glob.glob(os.path.join(
                model_root, *['*', "*.pt"])))[model_path]

        # Load defined model path
        else:
            model_name = model_path
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("Loaded model: {}".format(model_name))
