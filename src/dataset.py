import cv2
from glob import glob
import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np
import pandas as pd
from skimage.io import MultiImage
from src.utils import loadModel
from torch import nn
import torch

from Model import SegmentationModule, SAUNet, DualLoss

class ImageDataset(Dataset):
    def __init__(
            self, image_directory, mask_directory, labels_path, index_range,
            patch_size, number_of_patches, transform=None):
        data_frame = pd.read_csv(labels_path)[index_range[0]:index_range[1]]
        self.image_ids = np.array(
            data_frame["image_id"][data_frame["data_provider"] == "radboud"])
        self.image_ids = np.array(list(filter(
            lambda x: os.path.exists(os.path.join(
                mask_directory, x + "_mask.tiff")), self.image_ids)))
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.number_of_patches = number_of_patches
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, sample_index):
        tiff_image_filepath = os.path.join(
            self.image_directory, f"{self.image_ids[sample_index]}.tiff")
        tiff_mask_filepath = os.path.join(
            self.mask_directory, f"{self.image_ids[sample_index]}_mask.tiff")
        tiff_image = MultiImage(tiff_image_filepath)[2]
        tiff_mask = MultiImage(tiff_mask_filepath)[2]
        tile_patches = tile(
            self.number_of_patches, self.patch_size, tiff_image, tiff_mask)
        if self.transform:
            for sample in tile_patches:
                seed = random.randint(0, 2**32)
                random.seed(seed)
                sample["image"] = np.array(self.transform(Image.fromarray(
                    sample["image"])))
                random.seed(seed)
                sample["mask"] = np.array(self.transform(Image.fromarray(
                    sample["mask"])))
        for sample in tile_patches:
            sample["mask"] = sample["mask"][..., 0]
        concatenated_image, concatenated_mask = concatenatePatches(
            tile_patches)
        return \
            np.moveaxis(concatenated_image.astype(np.float32), -1, -3), \
            concatenated_mask.astype(np.float32)


class PredictedMaskDataset(ImageDataset):
    def __init__(
            self, image_directory, mask_directory, labels_path, index_range,
            patch_size, number_of_patches, transform=None):
        super(PredictedMaskDataset, self).__init__(
            image_directory, mask_directory, labels_path, index_range,
            patch_size, number_of_patches, transform)
        self.device = torch.device("cuda:0")
        self.model = SegmentationModule(
            DualLoss(6), SAUNet(6), 6).to(self.device)
        loadModel(self.model, "saved_models")

    def __getitem__(self, sample_index):
        image, _ = super(
            PredictedMaskDataset, self).__getitem__(sample_index)
        mask = torch.stack((self.model(image.to(self.device)),)*3)
        return mask


def concatenatePatches(tile_dictionary):
    N = len(tile_dictionary)
    col_size = int(np.sqrt(N))
    row_size = N // col_size
    patch_size = tile_dictionary[0]["image"].shape[0]
    concatenated_img = np.zeros((patch_size*col_size, patch_size*row_size, 3))
    concatenated_mask = np.zeros((patch_size*col_size, patch_size*row_size))
    for i, t in enumerate(tile_dictionary):
        img,mask,idx = t['image'],t['mask'],t['idx']
        x_1 = patch_size*(i%row_size)
        x_2 = patch_size*(i%row_size + 1)
        y_1 = patch_size*(i//row_size)
        y_2 = patch_size*(i//row_size + 1)
        concatenated_img[y_1:y_2, x_1:x_2] = img
        if mask is not None:
            concatenated_mask[y_1:y_2, x_1:x_2] = mask
    return concatenated_img, concatenated_mask


def tile(N, sz, img, mask=None):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if mask is not None:
        mask = np.pad(
            mask,[[pad0//2,pad0-pad0//2],
            [pad1//2,pad1-pad1//2],[0,0]],constant_values=0
        )
        mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
        mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
        if len(img) < N:
            mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    if mask is not None:
        mask = mask[idxs]
    for i in range(len(img)):
        result.append({
            "image": img[i],
            "mask": None if mask is None else mask[i],
            "idx": i})
    return result
