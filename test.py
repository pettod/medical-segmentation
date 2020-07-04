import cv2
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project files
from src.dataset import ImageDataset
from src.network import Net
from src.utils import loadModel

# Data paths
DATA_ROOT = os.path.realpath("../../REDS")
VALID_X_DIR = os.path.join(DATA_ROOT, "val_blur/")
VALID_Y_DIR = os.path.join(DATA_ROOT, "val_sharp/")

# Model parameters
MODEL_PATH = None
BATCH_SIZE = 4
PATCH_SIZE = 256
NUMBER_OF_DATALOADER_WORKERS = 8


def main():
    # Dataset
    valid_transforms = transforms.Compose([
        transforms.CenterCrop(PATCH_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])
    valid_dataset = ImageDataset(VALID_X_DIR, VALID_Y_DIR, valid_transforms)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUMBER_OF_DATALOADER_WORKERS)

    # Device (CPU / CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: Running on CPU")

    # Save directory
    save_directory = os.path.join(
        "predictions", time.strftime("%Y-%m-%d_%H%M%S"))
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # Predict and save
    with torch.no_grad():
        model = nn.DataParallel(Net()).to(device)
        loadModel(model, "saved_models", MODEL_PATH)
        for i, (X, y) in enumerate(tqdm(valid_dataloader)):
            X, y = X.to(device), y.numpy()
            output = model(X).cpu().numpy()
            X = X.cpu().numpy()
            for j in range(X.shape[0]):
                concat_image = np.moveaxis(np.concatenate(
                    [X[j], output[j], y[j]], axis=-1), 0, -1) * 255
                cv2.imwrite(os.path.join(
                    save_directory, f"{i}_{j}.png"), cv2.cvtColor(
                        concat_image.astype(np.uint8), cv2.COLOR_RGB2BGR))


main()
