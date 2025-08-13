'''
written by: Catherine Breen
June 2024

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682


'''
import colorsys
import torch
import cv2
import pandas as pd
import numpy as np
import tomli as tomllib
import utils
from torch.utils.data import Dataset, DataLoader
import IPython
import matplotlib.pyplot as plt
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from PIL import Image, ImageFile
import albumentations as A ### better for keypoint augmentations, pip install albumentations
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

class snowPoleDataset(Dataset):
    def __init__(self, data, sequences, keypoints, path, aug):
        self.data = data ## labels csv for metadata 
        self.sequences = sequences
        self.keypoints = keypoints
        self.path = path
        self.resize = 224
        
        if aug == False: 
            self.transform = A.Compose([
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy'))
        else: 
            self.transform = A.Compose([
                A.ToFloat(max_value=1.0),
                A.CropAndPad(px=50, p=1.0),  # Reduced padding for sequences
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=10, p=0.3),  # Reduced augmentation
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.3),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
                    A.ToGray(p=0.2)], p=0.3),
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy'))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence_data = self.sequences[index]
        keypoint_data = self.keypoints[index]
        filenames = []
        images = []
        keypoints_list = []
        
        for _, row in sequence_data.iterrows():
            # Load and process each image in the sequence
            filename = row["filename"]
            full_path = path_image_dict[filename]
            image = cv2.imread(str(full_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w, channel = image.shape
            image = image / 255.0
            image = cv2.resize(image, (self.resize, self.resize))

            # Process keypoints # it's a little bit redundant because we are processing the same keypoint 5 times 
            keypoints = keypoint_data #row[['x1','y1','x2','y2']]
            keypoints = keypoints.clip(lower=0)
            keypoints = np.array(keypoints, dtype='float32')
            keypoints = keypoints.reshape(-1, 2)
            keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
            #keypoints = keypoints / 224.0 ## add norm step because LSTM are initialized close to 0 

            #     # Debug the keypoint processing
            # print(f"Original keypoint_data: {keypoint_data}")
            # keypoints = keypoint_data #row[['x1','y1','x2','y2']]
            # print(f"Before clip: {keypoints}")
            # keypoints = keypoints.clip(lower=0)
            # keypoints = np.array(keypoints, dtype='float32')
            # print(f"After clip: {keypoints}")
            # keypoints = keypoints.reshape(-1, 2)
            # keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
            # print(f"After scaling: {keypoints}")

            # Apply transforms
            transformed = self.transform(image=image, keypoints= keypoints)
            img_transformed = transformed['image']
            keypoints_transformed = transformed['keypoints']
            images.append(np.transpose(img_transformed, (2, 0, 1)))
            keypoints_list.append(keypoints_transformed)
            filenames.append(filename)
        
        return {
            'image': torch.tensor(np.stack(images), dtype=torch.float),  # (seq_len, 3, 224, 224)
            'keypoints': torch.tensor(keypoints_list[-1], dtype=torch.float),  # Use last frame's keypoints
            'filenames':filenames, 
            'index': index  # Add this line
        }


###########################
## load data ##
# Load config from config.toml
with open("config_lstm.toml", "rb") as configfile:
    config = tomllib.load(configfile)

csv_path = f"{config['paths']['input_images']}/labels.csv"
image_path = config['paths']['input_images']
all_images = list(Path(image_path).rglob("*.JPG"))
path_image_dict = {img.name: img for img in all_images}

df_data = pd.read_csv(csv_path)
print(f'all rows in df_data {len(df_data.index)}')

## create sequences ## 
sequence_length = 3 
grouped = df_data.groupby(df_data['filename'].str.split('_').str[0])
X_sequences, keypoints = [], [] ## sequences (as filenames) and predictions (last filename's keypoint)
    
for camera_id, group in grouped:
    # Sort by timestamp/filename
    group_sorted = group.sort_values('filename')

    # Create overlapping sequences
    for i in range(len(group_sorted) - sequence_length + 1):
        seq_data = group_sorted.iloc[i:i + sequence_length]
        keypoint_prediction = group_sorted[['x1','y1','x2','y2']].iloc[i + sequence_length - 1] 
        X_sequences.append(seq_data)
        keypoints.append(keypoint_prediction)

### could add a random shuffle for cameras ## ? 

# Split into training and testing sets
train_size = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
y_train, y_test = keypoints[:train_size], keypoints[train_size:]

# initialize the dataset - `snowPoleDataset()`
train_data = snowPoleDataset(data = df_data, sequences = X_train, keypoints = y_train, path = path_image_dict, aug = False)
valid_data = snowPoleDataset(data = df_data, sequences = X_test, keypoints = y_test, path = path_image_dict, aug = False)

train_loader = DataLoader(
    train_data, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    valid_data, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0,
)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

if config["training"]["show_dataset_plot"]:
    utils.dataset_keypoints_plot(train_data)
    utils.dataset_keypoints_plot(valid_data)




