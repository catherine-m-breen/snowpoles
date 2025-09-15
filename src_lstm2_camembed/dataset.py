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
import json


def create_camera_mapping(df_data, output_path):
    """
    Create camera ID mapping from all samples and save it
    """
    # Extract camera IDs and years from filenames
    camera_ids = df_data["filename"].str.split("_").str[0]
    camera_years = df_data["year"]
    
    # Create camera+year combinations
    camera_year_combinations = camera_ids + "_" + camera_years.astype(str)
    unique_combinations = camera_year_combinations.unique()
    unique_combinations = sorted(unique_combinations)  # Sort for consistency
    
    # Create mapping: camera_year -> numeric_id
    ## assigns numbers starting at 0
    camera_mapping = {combo: idx for idx, combo in enumerate(unique_combinations)}
    
    print(f"Found {len(camera_mapping)} unique camera+year combinations:")
    for combo, idx in camera_mapping.items():
        print(f"  {combo}: {idx}")
    
    # Save mapping to file for later use (prediction, evaluation)
    os.makedirs(output_path, exist_ok=True)
    mapping_file = os.path.join(output_path, "camera_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(camera_mapping, f, indent=2) 

    print(f"Camera mapping saved to: {mapping_file}")
    return camera_mapping


class snowPoleDataset(Dataset):
    def __init__(self, data, sequences, keypoints, path_dict, aug, camera_mapping=None):
        self.data = data ## labels csv for metadata 
        self.sequences = sequences
        self.keypoints = keypoints
        self.path_dict = path_dict
        self.resize = 224
        self.camera_mapping = camera_mapping
        
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

        # camera_id_str =sequence_data['filename'].str.split('_').str[0].iloc[0]
        # camera_id_numeric = self.camera_mapping[camera_id_str]
        first_row = sequence_data.iloc[0]
        camera_id_str = first_row["filename"].split("_")[0]
        camera_year = str(first_row["year"])  # Make sure it's a string
        camera_year_combo = f"{camera_id_str}_{camera_year}"
        
        camera_id_numeric = self.camera_mapping[camera_year_combo]

        filenames = []
        images = []
        keypoints_list = []
        for _, row in sequence_data.iterrows():
            # Load and process each image in the sequence
            filename = row["filename"]
            #full_path = path_image_dict[filename]
            full_path = self.path_dict[filename]
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
            'camera_id': torch.tensor(camera_id_numeric, dtype=torch.long), ## dtype makes it a 64 bit integer ,
            'index': index  # Add this line
        }


###########################
## load data ##
# Load config from config.toml
with open("config_lstm.toml", "rb") as configfile:
    config = tomllib.load(configfile)

csv_path = f"{config['paths']['labels']}"
image_path = config['paths']['input_images']
all_images = list(Path(image_path).rglob("*.JPG"))
path_image_dict = {img.name: img for img in all_images}

df_data = pd.read_csv(csv_path)
print(f'all rows in df_data {len(df_data.index)}')

## create camera mapping ##
camera_mapping = create_camera_mapping(
    df_data,
    config['paths']['models_output']
)

# Make camera mapping globally available for other scripts
global_camera_mapping = camera_mapping
num_cameras = len(camera_mapping)


## create sequences ## 
sequence_length = 4
grouped = df_data.groupby(df_data['filename'].str.split('_').str[0])
X_sequences, keypoints = [], [] ## sequences (as filenames) and predictions (last filename's keypoint)
    
for camera_id, group in grouped:
    # Sort by timestamp/filename
    group_sorted = group.sort_values('filename')

    # Create overlapping sequences
    for i in range(len(group_sorted) - sequence_length + 1):
       # IPython.embed()
        seq_data = group_sorted.iloc[i:i + sequence_length]
        #keypoint_prediction = group_sorted[['x1','y1','x2','y2']].iloc[i + sequence_length - 1] 
        keypoint_prediction = group_sorted[['top_x','top_y','bottom_x','bottom_y']].iloc[i + sequence_length - 1] 
        X_sequences.append(seq_data)
        keypoints.append(keypoint_prediction)

### could add a random shuffle for cameras ## ? 

# Split into training and testing sets
train_size = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
y_train, y_test = keypoints[:train_size], keypoints[train_size:]

# initialize the dataset - `snowPoleDataset()`
train_data = snowPoleDataset(data = df_data, 
                             sequences = X_train, 
                             keypoints = y_train, 
                             path_dict = path_image_dict,
                             aug = False, 
                             camera_mapping=camera_mapping)

valid_data = snowPoleDataset(data = df_data, 
                             sequences = X_test, 
                             keypoints = y_test, 
                             path_dict = path_image_dict, 
                             aug = False, 
                             camera_mapping=camera_mapping)

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




