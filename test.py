import cv2
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import colors
import matplotlib.pyplot as plt

# Project files
from src.dataset import ImageDataset
from src.utils import loadModel
from Model import SegmentationModule, SAUNet, DualLoss


# Radboud clinic colors
RADBOUD_COLOR_CODES = {
    "0": np.array(["0 Background",   np.array([  0,   0,   0])]),
    "1": np.array(["1 Stroma",       np.array([153, 221, 255])]),
    "2": np.array(["2 Healthy",      np.array([  0, 153,  51])]),
    "3": np.array(["3 Gleason 3",    np.array([255, 209,  26])]),
    "4": np.array(["4 Gleason 4",    np.array([255, 102,   0])]),
    "5": np.array(["5 Gleason 5",    np.array([255,   0,   0])]),
}

# Data paths
DATA_ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
IMAGE_DIR = os.path.join(DATA_ROOT, "train_images")
MASK_DIR = os.path.join(DATA_ROOT, "train_label_masks")
LABELS_PATH = os.path.join(DATA_ROOT, "train.csv")
INDEX_SPLIT = 9554

# Model parameters
PATCH_SIZE = 64
NUMBER_OF_PATCHES = 16
MODEL_PATH = None
DROP_LAST_BATCH = True


def colorizeMask(mask):
    r = np.copy(mask)
    g = np.copy(mask)
    b = np.copy(mask)
    for i in range(len(RADBOUD_COLOR_CODES)):
        r[r == i] = RADBOUD_COLOR_CODES[str(i)][1][0] / 255
        g[g == i] = RADBOUD_COLOR_CODES[str(i)][1][1] / 255
        b[b == i] = RADBOUD_COLOR_CODES[str(i)][1][2] / 255
    mask = cv2.merge((r, g, b))
    return mask


def main():
    # Create Pytorch generator
    dataset = ImageDataset(
        IMAGE_DIR, MASK_DIR, LABELS_PATH, [INDEX_SPLIT, -1], PATCH_SIZE,
        NUMBER_OF_PATCHES)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)

    # Load model
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(SegmentationModule(
        DualLoss(6), SAUNet(6), 6)).to(device)
    loadModel(model, "saved_models")

    # Color bar details
    cmap = colors.ListedColormap(
        list(np.array(list(RADBOUD_COLOR_CODES.values()))[:, 1] / 255))
    grades = list(np.arange(0, 13))
    grades_descriptions = [""] * 13
    grades_descriptions[1::2] = list(np.array(list(
        RADBOUD_COLOR_CODES.values()))[:, 0])
    norm = colors.BoundaryNorm(grades, cmap.N+1)

    # Load batch
    for image_batch, mask_batch in dataloader:
        prediction = model(image_batch.to(device), segSize=True)
        image = image_batch.numpy()[0]
        if image.shape[0] == 3:
            image = np.moveaxis(image, -3, -1)
        image /= 255
        gt_mask = mask_batch.numpy()[0]
        prediction_mask = prediction.cpu().numpy()[0].astype(np.float32)
        gt_mask = colorizeMask(gt_mask)
        prediction_mask = colorizeMask(prediction_mask)

        # Plot mask and image
        plotted_cell_mask = plt.imshow(
            cv2.hconcat([image, prediction_mask, gt_mask]),
            cmap=cmap, norm=norm)
        colorbar = plt.colorbar(plotted_cell_mask, cmap=cmap, ticks=grades)
        colorbar.ax.set_yticklabels(grades_descriptions)
        plt.draw()
        plt.pause(4)
        plt.clf()


with torch.no_grad():
    main()
