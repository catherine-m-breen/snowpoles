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


# Load config from config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)

def apply_filter(image):
    # width, height, __ = image.shape
    # for y in range(height):
    #     for x in range(width):
    #         pixel = list(colorsys.rgb_to_hsv(*image[x, y]))
    #         if (pixel[0] < 0.833):
    #             image[x, y] = (0, 0, 0)
    #             continue
    #         pixel[1] = 1
    #         pixel[2] = 255
    #         rgb = colorsys.hsv_to_rgb(*pixel)
    #         image[x, y] = (round(rgb[0]), round(rgb[1]), round(rgb[2]))
    image_rgb = image[:, :, ::-1]
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = image_hsv[:, :, 0] < 149
    image_rgb[mask] = [0,0,0]
    image_hsv[~mask, 1] = 255
    image_hsv[~mask, 2] = 255
    valid_pixels = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    image_rgb[~mask] = valid_pixels[~mask]
    #print("filtered applied!")
    return image_rgb[:, :, ::-1]

    

# Define a function to sample every third photo
## Only used for experiments 
def sample_every_x(group, x):
    indices = np.arange(len(group[1]))
    every_x = len(group[1])//x
    selected_indices = indices[2::every_x]  
    return group[1].iloc[selected_indices]

def create_camera_mapping(training_samples, valid_samples, output_path):
    """
    Create camera ID mapping from all samples and save it
    """
    # Extract camera IDs from all filenames
    all_samples = pd.concat([training_samples, valid_samples], ignore_index=True)
    camera_ids = all_samples["filename"].str.split("_").str[0].unique()
    camera_ids = sorted(camera_ids)  # Sort for consistency
    
    # Create mapping: camera_name -> numeric_id
    ## assigns numbers starting at 0
    camera_mapping = {camera: idx for idx, camera in enumerate(camera_ids)}
    
    print(f"Found {len(camera_mapping)} unique cameras:")
    for camera, idx in camera_mapping.items():
        print(f"  {camera}: {idx}")
    
    # Save mapping to file for later use (prediction, evaluation)
    mapping_file = os.path.join(output_path, "camera_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(camera_mapping, f, indent=2) 

    print(f"Camera mapping saved to: {mapping_file}")
    return camera_mapping

def train_test_split(csv_path, image_path):

    df_data = pd.read_csv(csv_path)
    print(f'all rows in df_data {len(df_data.index)}')

    training_samples = df_data.sample(frac=0.8, random_state=100) # same shuffle everytime
    valid_samples = df_data[~df_data.index.isin(training_samples.index)]

    ## check to make sure we only use images that exist
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

    # save labels to output folder
    if not os.path.exists(f"{config['paths']['models_output']}"):
        os.makedirs(f"{config['paths']['models_output']}", exist_ok=True)
    training_samples.to_csv(f"{config['paths']['models_output']}/training_samples.csv")
    valid_samples.to_csv(f"{config['paths']['models_output']}/valid_samples.csv")

    print(f'# of examples we will now train on {len(training_samples)}, val on {len(valid_samples)}')

    return training_samples, valid_samples

class snowPoleDataset(Dataset):

    def __init__(self, samples, path, aug, camera_mapping=None): # split='train'):
        self.data = samples
        self.path = path
        self.resize = 224
        self.camera_mapping = camera_mapping

        if aug == False: 
            self.transform = A.Compose([
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy'))
        else: 
            self.transform = A.Compose([
                A.ToFloat(max_value=1.0),
                A.CropAndPad(px=75, p =1.0), ## final model is 50 pixels
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
        #IPython.embed()
        cameraID = self.data.iloc[index]["filename"].split("_")[0]  
        filename = self.data.iloc[index]["filename"]
        datetime_str = self.data.iloc[index]['datetime']

        camera_id_numeric = self.camera_mapping[cameraID]

        image = cv2.imread(parents[self.data.iloc[index]["filename"]])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        image = image / 255.0
        
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        #IPython.embed()
        if config['training']['filter']: 
            image = apply_filter(image)
            if index % 100: 
                cv2.imwrite(f"{config['paths']['models_output']}/filtered_{filename}", image)
        #image = image / 255.0
        # get the keypoints
        keypoints = self.data.iloc[index][1:][['x1','y1','x2','y2']]  #[3:7]  ### change to x1 y1 x2 y2
        
        # adonis neg values # 
        keypoints = keypoints.clip(lower=0)

        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)

        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]

        transformed = self.transform(image=image, keypoints=keypoints)
        img_transformed = transformed['image']
        keypoints = transformed['keypoints']

        # viz training data

        #utils.vis_keypoints(transformed['image'], transformed['keypoints'])
        image = np.transpose(img_transformed, (2, 0, 1))

        if len(keypoints) != 2:
            utils.vis_keypoints(transformed['image'], transformed['keypoints'])

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
            'camera_id': torch.tensor(camera_id_numeric, dtype=torch.long), ## dtype makes it a 64 bit integer 
            'filename': filename,
            'datetime': datetime_str
        }

# get the training and validation data samples
training_samples, valid_samples = train_test_split(
    f"{config['paths']['labels']}", config['paths']['input_images']
)

# CREATE CAMERA MAPPING
camera_mapping = create_camera_mapping(
    training_samples, 
    valid_samples, 
    config['paths']['models_output']
)

# Make camera mapping globally available for other scripts
global_camera_mapping = camera_mapping
num_cameras = len(camera_mapping)


#{config['paths']['input_images']}/

# initialize the dataset - `snowPoleDataset()`
train_data = snowPoleDataset(
    training_samples,
    f"{config['paths']['input_images']}",
    aug=config['training']['aug'],
    camera_mapping = camera_mapping
)  ## we want all folders

valid_data = snowPoleDataset(
    valid_samples, f"{config['paths']['input_images']}", aug=False,
    camera_mapping = camera_mapping
)  # we always want the transform to be the normal transform

# prepare data loaders
train_loader = DataLoader(
    train_data, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    valid_data,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=0,
)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

if config["training"]["show_dataset_plot"]:
    utils.dataset_keypoints_plot(train_data)
    utils.dataset_keypoints_plot(valid_data)




