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


# Load config from config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)

def apply_filter(image):
    image_rgb = image[:, :, ::-1]
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = image_hsv[:, :, 0] < 149
    image_rgb[mask] = [0,0,0]
    image_hsv[~mask, 1] = 255
    image_hsv[~mask, 2] = 255
    valid_pixels = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    image_rgb[~mask] = valid_pixels[~mask]
    return image_rgb[:, :, ::-1]

def sample_every_x(group, x):
    indices = np.arange(len(group[1]))
    every_x = len(group[1])//x
    selected_indices = indices[2::every_x]  
    return group[1].iloc[selected_indices]

def train_test_split(csv_path, image_path):
    df_data = pd.read_csv(csv_path)
    print(f'all rows in df_data {len(df_data.index)}')

    training_samples = df_data.sample(frac=0.8, random_state=100)
    valid_samples = df_data[~df_data.index.isin(training_samples.index)]

    all_images = list(Path(image_path).rglob("*.JPG"))
    
    global parents
    parents = {}
    for i in all_images:
        parents[i.name] = str(i)
    filenames = [img.name for img in all_images]
    valid_samples = valid_samples[
        valid_samples["filename"].isin(filenames)
    ].reset_index()
    training_samples = training_samples[
        training_samples["filename"].isin(filenames)
    ].reset_index()

    if not os.path.exists(f"{config['paths']['models_output']}"):
        os.makedirs(f"{config['paths']['models_output']}", exist_ok=True)
    training_samples.to_csv(f"{config['paths']['models_output']}/training_samples.csv")
    valid_samples.to_csv(f"{config['paths']['models_output']}/valid_samples.csv")

    print(f'# of examples we will now train on {len(training_samples)}, val on {len(valid_samples)}')
    return training_samples, valid_samples

class snowPoleDataset(Dataset):
    """Original dataset for single images (CNN training)"""
    
    def __init__(self, samples, path, aug):
        self.data = samples
        self.path = path
        self.resize = 224

        if aug == False: 
            self.transform = A.Compose([
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy'))
        else: 
            self.transform = A.Compose([
                A.ToFloat(max_value=1.0),
                A.CropAndPad(px=75, p =1.0),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=20, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.ToGray(p=0.5)], p = 0.5),
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.data)

    def __filename__(self, index):
        filename = self.data.iloc[index]['filename']
        return filename
    
    def __getitem__(self, index):
        cameraID = self.data.iloc[index]["filename"].split("_")[0]  
        filename = self.data.iloc[index]["filename"]

        image = cv2.imread(parents[self.data.iloc[index]["filename"]])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        image = image / 255.0
        
        image = cv2.resize(image, (self.resize, self.resize))
        
        if config['training']['filter']: 
            image = apply_filter(image)
            if index % 100: 
                cv2.imwrite(f"{config['paths']['models_output']}/filtered_{filename}", image)

        keypoints = self.data.iloc[index][1:][['x1','y1','x2','y2']]
        keypoints = keypoints.clip(lower=0)
        keypoints = np.array(keypoints, dtype='float32')
        keypoints = keypoints.reshape(-1, 2)
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]

        transformed = self.transform(image=image, keypoints=keypoints)
        img_transformed = transformed['image']
        keypoints = transformed['keypoints']

        image = np.transpose(img_transformed, (2, 0, 1))

        if len(keypoints) != 2:
            utils.vis_keypoints(transformed['image'], transformed['keypoints'])

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
            'filename': filename
        }

class snowPoleSequenceDataset(Dataset):
    """New dataset for sequences (LSTM training)"""
    
    def __init__(self, samples, path, sequence_length=5, aug=False):
        self.data = samples
        self.path = path
        self.sequence_length = sequence_length
        self.resize = 224
        
        # Group data by camera and sort by datetime
        self.data['datetime'] = pd.to_datetime(self.data['datetime'], errors='coerce')
        self.grouped_data = self.data.groupby('cameraID').apply(
            lambda x: x.sort_values('datetime')
        ).reset_index(drop=True)
        
        # Create valid sequence indices
        self.valid_sequences = self._create_sequence_indices()
        
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

    def _create_sequence_indices(self):
        """Create indices for valid sequences"""
        valid_sequences = []

        for camera_id in self.grouped_data['cameraID'].unique():
            camera_data = self.grouped_data[self.grouped_data['cameraID'] == camera_id]
            camera_indices = camera_data.index.tolist()
            
            # Create sequences with overlap
            for i in range(len(camera_indices) - self.sequence_length + 1):
                sequence_indices = camera_indices[i:i + self.sequence_length]
                # Use the last frame's keypoints as target
                target_idx = sequence_indices[-1]

                           # Get filenames for the sequence
                sequence_filenames = [self.grouped_data.loc[idx, 'filename'] for idx in sequence_indices]
                valid_sequences.append({
                    'sequence_indices': sequence_indices,
                    'target_idx': target_idx,
                    'camera_id': camera_id,
                    'filenames': sequence_filenames
                })
        
        IPython.embed()
        return valid_sequences

    def __len__(self):
        return len(self.valid_sequences)

    def _load_and_process_image(self, filename):
        """Load and process a single image"""
        try:
            image = cv2.imread(parents[filename])
            if image is None:
                # Return a black image if file not found
                return np.zeros((self.resize, self.resize, 3), dtype=np.float32)
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w, channel = image.shape
            image = image / 255.0
            image = cv2.resize(image, (self.resize, self.resize))
            
            if config['training']['filter']: 
                image = apply_filter(image)
                
            return image, orig_h, orig_w
        except:
            # Return a black image if any error occurs
            return np.zeros((self.resize, self.resize, 3), dtype=np.float32), self.resize, self.resize

    def __getitem__(self, index):
        sequence_info = self.valid_sequences[index]
        sequence_indices = sequence_info['sequence_indices']
        target_idx = sequence_info['target_idx']
        
        # Load sequence of images
        sequence_images = []
        sequence_keypoints = []
        
        for seq_idx in sequence_indices:
            row = self.grouped_data.iloc[seq_idx]
            filename = row["filename"]
            
            # Load image
            if isinstance(self._load_and_process_image(filename), tuple):
                image, orig_h, orig_w = self._load_and_process_image(filename)
            else:
                image = self._load_and_process_image(filename)
                orig_h, orig_w = self.resize, self.resize
            
            # Get keypoints
            keypoints = row[['x1','y1','x2','y2']].values
            keypoints = np.clip(keypoints, 0, None)  # Remove negative values
            keypoints = keypoints.astype('float32').reshape(-1, 2)
            
            # Scale keypoints
            keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
            
            # Apply same transform to all images in sequence for consistency
            if len(sequence_images) == 0:  # Apply transform only to first image, then use same params
                transformed = self.transform(image=image, keypoints=keypoints)
                self.current_transform_params = transformed
            else:
                # For sequence consistency, apply minimal transform
                transformed = A.Compose([A.Resize(224, 224)], 
                                      keypoint_params=A.KeypointParams(format='xy'))(
                                          image=image, keypoints=keypoints)
            
            img_transformed = transformed['image']
            keypoints_transformed = transformed['keypoints']
            
            # Convert to tensor format
            img_tensor = np.transpose(img_transformed, (2, 0, 1))
            sequence_images.append(torch.tensor(img_tensor, dtype=torch.float))
            sequence_keypoints.append(torch.tensor(keypoints_transformed, dtype=torch.float))
        
        # Stack sequence
        sequence_tensor = torch.stack(sequence_images)  # (seq_len, C, H, W)
        
        # Use target keypoints (last frame)
        target_keypoints = sequence_keypoints[-1]
        
        return {
            'sequence': sequence_tensor,
            'keypoints': target_keypoints,
            'filename': self.grouped_data.iloc[target_idx]["filename"],
            'camera_id': sequence_info['camera_id']
        }

# Create datasets based on config
training_samples, valid_samples = train_test_split(
    f"{config['paths']['input_images']}/labels.csv", config['paths']['input_images']
)

# Standard single-image datasets
train_data = snowPoleDataset(
    training_samples,
    f"{config['paths']['input_images']}",
    aug=config['training']['aug'],
)

valid_data = snowPoleDataset(
    valid_samples, 
    f"{config['paths']['input_images']}", 
    aug=False
)

# Sequence datasets for LSTM
sequence_length = config.get('training', {}).get('sequence_length', 5)

train_data_sequence = snowPoleSequenceDataset(
    training_samples,
    f"{config['paths']['input_images']}",
    sequence_length=sequence_length,
    aug=config['training']['aug'],
)

valid_data_sequence = snowPoleSequenceDataset(
    valid_samples,
    f"{config['paths']['input_images']}",
    sequence_length=sequence_length,
    aug=False,
)

# Standard data loaders
train_loader = DataLoader(
    train_data, 
    batch_size=config['training']['batch_size'], 
    shuffle=True, 
    num_workers=0
)

valid_loader = DataLoader(
    valid_data,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=0,
)

# Sequence data loaders
train_loader_sequence = DataLoader(
    train_data_sequence, 
    batch_size=max(1, config['training']['batch_size'] // 2),  # Smaller batch size for sequences
    shuffle=True, 
    num_workers=0
)

valid_loader_sequence = DataLoader(
    valid_data_sequence,
    batch_size=max(1, config['training']['batch_size'] // 2),
    shuffle=False,
    num_workers=0,
)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")
print(f"Training sequence instances: {len(train_data_sequence)}")
print(f"Validation sequence instances: {len(valid_data_sequence)}")

if config["training"]["show_dataset_plot"]:
    utils.dataset_keypoints_plot(train_data)
    utils.dataset_keypoints_plot(valid_data)