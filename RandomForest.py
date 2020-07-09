import gc
import glob
import os
import shutil
import signal
import sys
import time
from collections import Counter, OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from torch import nn
import torch
import lightgbm as lgb
from Model.models import SAUNet, SegmentationModule
from Model.loss import DualLoss
from src.dataset import concatenatePatches, tile
from src.utils import loadModel
INDEX_SPLIT = 9554
def normalizeArray(data_array, max_value=255):
    return ((data_array / max_value - 0.5) * 2).astype(np.float32)
def unnormalizeArray(data_array, max_value=255):
    data_array = (data_array / 2 + 0.5) * max_value
    data_array[data_array < 0.0] = 0.0
    data_array[data_array > max_value] = max_value
    return data_array.astype(np.uint8)
# Model parameters
BATCH_SIZE = 4
PATCH_SIZE = 64
NUMBER_OF_PATCHES = 16

LOAD_MODEL = True
MODEL_PATH = None
PATIENCE = 10
LEARNING_RATE = 1e-4
DROP_LAST_BATCH = True
NUMBER_OF_DATALOADER_WORKERS = 2

DATA = "/media/ubuntu/2tb_hdd/prostate-cancer-grade-assessment"
gls2isu = {"0+0":0,'negative':0,'3+3':1,'3+4':2,'4+3':3,'4+4':4,'3+5':4,'5+3':4,'4+5':5,'5+4':5,'5+5':5}

df_train = pd.read_csv(os.path.join(DATA, "train.csv"))
# df_train = df_train[df_train.data_provider == 'radboud']
def extract_features(mask):
    counts = []
    for i in range(1,6):
        counts.append(np.count_nonzero(mask == i))
    percents = np.array(counts).astype(np.float32)
    percents /= percents.sum()
    return counts, percents

def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')

for label in range(1,6):
    df_train[f'percent_{label}'] = None
    df_train[f'count_{label}'] = None

model = nn.DataParallel(SegmentationModule(DualLoss(6), SAUNet(6), 6)).cuda()
loadModel(model, "saved_models")

for i in tqdm(range(len(df_train))):
    idx = df_train.iloc[i, 0]
    isup = df_train.iloc[i, 2]
    gleason = df_train.iloc[i, 3]
    image_file = os.path.join(DATA, 'train_images', f'{idx}.tiff')
    image = skimage.io.MultiImage(image_file)
    image = np.array(image[2]) # middle resolution
    image = concatenatePatches(tile(16, PATCH_SIZE, image))[0]
    image = image / 255 # ormalizeArray(image)
    image = np.moveaxis(image.astype(np.float32), -1, -3)
    image = torch.Tensor(image).cuda()
    image = torch.stack([image, image], dim=0)
    mask = model(image, segSize=True).cpu().numpy()[0]
    # Radboud clinic colors
    RADBOUD_COLOR_CODES = {
        "0": np.array(["0 Background",   np.array([  0,   0,   0])]),
        "1": np.array(["1 Stroma",       np.array([153, 221, 255])]),
        "2": np.array(["2 Healthy",      np.array([  0, 153,  51])]),
        "3": np.array(["3 Gleason 3",    np.array([255, 209,  26])]),
        "4": np.array(["4 Gleason 4",    np.array([255, 102,   0])]),
        "5": np.array(["5 Gleason 5",    np.array([255,   0,   0])]),
    }

    # Color bar details
    from matplotlib import colors
    cmap = colors.ListedColormap(
        list(np.array(list(RADBOUD_COLOR_CODES.values()))[:, 1] / 255))
    grades = list(np.arange(0, 13))
    grades_descriptions = [""] * 13
    grades_descriptions[1::2] = list(np.array(list(
        RADBOUD_COLOR_CODES.values()))[:, 0])
    norm = colors.BoundaryNorm(grades, cmap.N+1)

    # Colorize mask
    image = np.moveaxis(image[0].cpu().numpy(), -3, -1)
    mask = mask.astype(np.float32)
    r = np.copy(mask)
    g = np.copy(mask)
    b = np.copy(mask)
    for i in range(len(RADBOUD_COLOR_CODES)):
        r[r == i] = RADBOUD_COLOR_CODES[str(i)][1][0]/255
        g[g == i] = RADBOUD_COLOR_CODES[str(i)][1][1]/255
        b[b == i] = RADBOUD_COLOR_CODES[str(i)][1][2]/255
    mask = cv2.merge((r, g, b))
    # image /= 255
    # Plot mask and image
    plotted_cell_mask = plt.imshow(
        cv2.hconcat([image, mask]), cmap=cmap, norm=norm)
    colorbar = plt.colorbar(plotted_cell_mask, cmap=cmap, ticks=grades)
    colorbar.ax.set_yticklabels(grades_descriptions)
    plt.draw()
    plt.pause(2)
    plt.clf()

    cnt, feat = extract_features(mask)
    for label in range(1,6):
        df_train[f'count_{label}'].iloc[i] = cnt[label-1]
        df_train[f'percent_{label}'].iloc[i] = feat[label-1]
    # if os.path.exists(mask_file):
    #     mask = skimage.io.MultiImage(mask_file)
    #     mask = np.array(mask[1]) # middle resolution
    #     cnt, feat = extract_features(mask)
    #     for label in range(1,6):
    #         df_train[f'count_{label}'].iloc[i] = cnt[label-1]
    #         df_train[f'percent_{label}'].iloc[i] = feat[label-1]
    # else:
    #     continue

print(df_train)
df_train = df_train.replace(to_replace='None', value=np.nan).dropna()
df_train.reset_index(drop=True)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
splits = list(skf.split(df_train, df_train.isup_grade))

#features = [f"percent_{label}" for label in range(1, 6)] 
features = [f"percent_{label}" for label in range(1, 6)] + [f"count_{label}" for label in range(1, 6)]
target = 'isup_grade'


scores = []
for fold in range(5):
    train = df_train.iloc[splits[fold][0]]
    valid = df_train.iloc[splits[fold][1]]
    
    model = KNeighborsClassifier(n_neighbors=5)
    
    model.fit(train[features], train[target])
    
    preds = model.predict(valid[features])
    
    score = quadratic_weighted_kappa(preds, valid[target])
    scores.append(score)
    print(f"Fold = {fold}, QWK = {score:.4f}")
    
print(f"Mean = {np.mean(scores):.4f}")


scores = []
for fold in range(5):
    train = df_train.iloc[splits[fold][0]]
    valid = df_train.iloc[splits[fold][1]]
    
    model = RandomForestClassifier(random_state=42)
    
    model.fit(train[features], train[target])
    
    preds = model.predict(valid[features])
    
    score = quadratic_weighted_kappa(preds, valid[target])
    scores.append(score)
    print(f"Fold = {fold}, QWK = {score:.4f}")
    
print(f"Mean = {np.mean(scores):.4f}")


def QWK(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.rint(preds)
    score = quadratic_weighted_kappa(preds, labels)
    return ("QWK", score, True)

scores = []
for fold in range(5):
    train = df_train.iloc[splits[fold][0]]
    valid = df_train.iloc[splits[fold][1]]
    
    train_dataset = lgb.Dataset(train[features], train[target])
    valid_dataset = lgb.Dataset(valid[features], valid[target])
    
    params = {
                "objective": 'regression',
                "metric": 'rmse',
                "seed": 42,
                "learning_rate": 0.01,
                "boosting": "gbdt",
            }
        
    model = lgb.train(
                params=params,
                num_boost_round=1000,
                early_stopping_rounds=200,
                train_set=train_dataset,
                valid_sets=[train_dataset, valid_dataset],
                verbose_eval=100,
                feval=QWK,
            )
        
    
    preds = model.predict(valid[features], num_iteration=model.best_iteration)
    preds = np.rint(preds)
    
    score = quadratic_weighted_kappa(preds, valid[target])
    scores.append(score)
    
    print(f"Fold = {fold}, QWK = {score:.4f}")
    
print(f"Mean = {np.mean(scores):.4f}")
