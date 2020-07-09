import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import os
import time
from tqdm import trange
import numpy as np
import cv2

# Project files
from Model import SegmentationModule, SAUNet, DualLoss
from src.callbacks import CsvLogger, EarlyStopping
from src.utils import \
    initializeEpochMetrics, updateEpochMetrics, getProgressbarText, \
    saveLearningCurve, loadModel


class Learner():
    def __init__(
            self, train_dataset, valid_dataset, batch_size, learning_rate,
            patience, num_workers=1, load_pretrained_weights=False,
            model_path=None, drop_last_batch=False):
        # Device (CPU / CUDA)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: Running on CPU\n\n\n\n")

        # Model, optimizer
        self.model_root = "saved_models"
        save_model_directory = os.path.join(
            self.model_root, time.strftime("%Y-%m-%d_%H%M%S"))
        self.model = nn.DataParallel(SegmentationModule(
            DualLoss(6), SAUNet(6), 6)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        loadModel(
            self.model, self.model_root, model_path, self.optimizer,
            load_pretrained_weights)

        # Callbacks, scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", 0.3, patience//3, min_lr=1e-8)
        self.csv_logger = CsvLogger(save_model_directory)
        self.early_stopping = EarlyStopping(save_model_directory, patience)
        self.epoch_metrics = {}

        # Train and validation batch generators
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=drop_last_batch)
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=drop_last_batch)
        self.number_of_train_batches = len(self.train_dataloader)
        self.number_of_valid_batches = len(self.valid_dataloader)

    def validationEpoch(self):
        print()
        progress_bar = trange(self.number_of_valid_batches, leave=False)
        progress_bar.set_description(" Validation")

        # Run batches
        for i, (X, y) in zip(progress_bar, self.valid_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            prediction, loss = self.predict(X, y, "train")
            updateEpochMetrics(
                prediction, y, loss, i, self.epoch_metrics, "valid",
                self.optimizer)

        # Logging
        print("\n{}".format(getProgressbarText(self.epoch_metrics, "Valid")))
        self.csv_logger.__call__(self.epoch_metrics)
        validation_loss = self.epoch_metrics["valid_loss"]
        self.early_stopping.__call__(
            validation_loss, self.model, self.optimizer)
        self.scheduler.step(validation_loss)
        saveLearningCurve(model_root=self.model_root)

    def train(self):
        # Run epochs
        epochs = 1000
        for epoch in range(1, epochs+1):
            if self.early_stopping.isEarlyStop():
                break
            progress_bar = trange(self.number_of_train_batches, leave=False)
            progress_bar.set_description(f" Epoch {epoch}/{epochs}")
            self.epoch_metrics = initializeEpochMetrics(epoch)

            # Run batches
            for i, (X, y) in zip(progress_bar, self.train_dataloader):

                # Validation epoch before last batch
                if i == self.number_of_train_batches - 1:
                    with torch.no_grad(): 
                        self.validationEpoch()

                # Feed forward and backpropagation
                X, y = X.to(self.device), y.to(self.device)
                self.model.zero_grad()
                prediction, loss = self.predict(X, y, "train")
                loss.backward()
                self.optimizer.step()

                # Compute metrics
                with torch.no_grad():
                    updateEpochMetrics(
                        prediction, y, loss, i, self.epoch_metrics, "train")
                    progress_bar.display(
                        getProgressbarText(self.epoch_metrics, "Train"), 1)

    def predict(self, x, y, mode="train"):
        if mode == "test":
            prediction = self.model({
                "image": x,
                "name": "a"},
                segSize=True
            )
            return prediction
        elif mode == "train" or mode == "valid":
            canny = np.array(
                [
                    cv2.Canny(np.uint8(x_), 10, 100) 
                    for x_ in list(np.moveaxis(x.cpu().numpy(), -3, -1))
                ]
            )
            prediction, loss, acc = self.model({
                "image": x,
                "mask": (
                    y, 
                    torch.from_numpy(canny).to(self.device).float()
                ),
                "name": "a"}
            )
            return prediction, loss.mean()
