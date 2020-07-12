import os
from torchvision import transforms

# Project files
from src.dataset import ImageDataset
from src.learner import Learner

# Data paths
DATA_ROOT = os.path.realpath("../input/prostate-cancer-grade-assessment")
IMAGE_DIR = os.path.join(DATA_ROOT, "train_images")
MASK_DIR = os.path.join(DATA_ROOT, "train_label_masks")
LABELS_PATH = os.path.join(DATA_ROOT, "train.csv")
INDEX_SPLIT = 9554

# Model parameters
BATCH_SIZE = 8
PATCH_SIZE = 64
NUMBER_OF_PATCHES = 16

LOAD_MODEL = False
MODEL_PATH = None
PATIENCE = 10
LEARNING_RATE = 1e-4
DROP_LAST_BATCH = True
NUMBER_OF_DATALOADER_WORKERS = 2


def main():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    valid_transforms = transforms.Compose([
    ])
    train_dataset = ImageDataset(
        IMAGE_DIR, MASK_DIR, LABELS_PATH, [0, INDEX_SPLIT], PATCH_SIZE,
        NUMBER_OF_PATCHES, train_transforms)
    valid_dataset = ImageDataset(
        IMAGE_DIR, MASK_DIR, LABELS_PATH, [INDEX_SPLIT, -1], PATCH_SIZE,
        NUMBER_OF_PATCHES, valid_transforms)
    learner = Learner(
        train_dataset, valid_dataset, BATCH_SIZE, LEARNING_RATE,
        PATIENCE, NUMBER_OF_DATALOADER_WORKERS, LOAD_MODEL, MODEL_PATH,
        DROP_LAST_BATCH)
    learner.train()


main()
