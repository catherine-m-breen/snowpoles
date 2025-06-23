"""
written by: Catherine Breen
June 2024

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682


"""

import torch
import cv2
import pandas as pd
import numpy as np
import tomllib

import utils
from torch.utils.data import Dataset, DataLoader
import IPython
import matplotlib.pyplot as plt
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from PIL import Image, ImageFile
import albumentations as A  ### better for keypoint augmentations, pip install albumentations
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# Comment out this line to disable dark mode
plt.style.use("./themes/dark.mplstyle")

# Load config from config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)

# Load config from config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)



# Define a function to sample every third photo
## Only used for experiments
def sample_every_x(group, x):
    indices = np.arange(len(group[1]))
    every_x = len(group[1]) // x
    selected_indices = indices[2::every_x]
    return group[1].iloc[selected_indices]


def train_test_split(csv_path, image_path):

    df_data = pd.read_csv(csv_path)
    print(f"all rows in df_data {len(df_data.index)}")

    training_samples = df_data.sample(
        frac=0.8, random_state=100
    )  # same shuffle everytime
    valid_samples = df_data[~df_data.index.isin(training_samples.index)]

    ## check to make sure we only use images that exist
    all_images = list(Path(image_path).rglob("*.JPG"))
    global parents
    parents = {}
    for i in all_images:
        parents[str(i).split("/")[-1]] = str(i)
    filenames = [img.name for img in all_images]
    valid_samples = valid_samples[
        valid_samples["filename"].isin(filenames)
    ].reset_index()
    training_samples = training_samples[
        training_samples["filename"].isin(filenames)
    ].reset_index()

    # save labels to output folder
    if not os.path.exists(f"{config["paths"]["models_output"]}"):
        os.makedirs(f"{config["paths"]["models_output"]}", exist_ok=True)
    training_samples.to_csv(f"{config["paths"]["models_output"]}/training_samples.csv")
    valid_samples.to_csv(f"{config["paths"]["models_output"]}/valid_samples.csv")

    print(
        f"# of examples we will now train on {len(training_samples)}, val on {len(valid_samples)}"
    )

    return training_samples, valid_samples


class snowPoleDataset(Dataset):

    def __init__(self, samples, path, aug):  # split='train'):
        self.data = samples
        self.path = path
        self.resize = 224

        if aug == False:
            self.transform = A.Compose(
                [
                    A.Resize(224, 224),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        else:
            self.transform = A.Compose(
                [
                    A.ToFloat(max_value=1.0),
                    A.CropAndPad(px=75, p=1.0),  ## final model is 50 pixels
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.2, rotate_limit=20, p=0.5
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=0.5),
                            A.ColorJitter(
                                brightness=0.2,
                                contrast=0.2,
                                saturation=0.2,
                                hue=0.2,
                                always_apply=False,
                                p=0.5,
                            ),
                            A.ToGray(p=0.5),
                        ],
                        p=0.5,
                    ),
                    A.Resize(224, 224),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )

    def __len__(self):
        return len(self.data)

    def __filename__(self, index):

        filename = self.data.iloc[index]["filename"]
        return filename

    def __getitem__(self, index):
        cameraID = self.data.iloc[index]["filename"].split("_")[
            0
        ]  ## may need to update this. *Yeah, you think?* -Nesitive
        filename = self.data.iloc[index]["filename"]

        image = cv2.imread(parents[self.data.iloc[index]["filename"]])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape

        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        image = image / 255.0
        # get the keypoints
        keypoints = self.data.iloc[index][1:][
            ["x1", "y1", "x2", "y2"]
        ]  # [3:7]  ### change to x1 y1 x2 y2
        keypoints = np.array(keypoints, dtype="float32")
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)

        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]

        transformed = self.transform(image=image, keypoints=keypoints)
        img_transformed = transformed["image"]
        keypoints = transformed["keypoints"]

        # viz training data

        # utils.vis_keypoints(transformed['image'], transformed['keypoints'])
        image = np.transpose(img_transformed, (2, 0, 1))

        if len(keypoints) != 2:
            utils.vis_keypoints(transformed["image"], transformed["keypoints"])

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "keypoints": torch.tensor(keypoints, dtype=torch.float),
            "filename": filename,
        }


# get the training and validation data samples
training_samples, valid_samples = train_test_split(
    f"{config["paths"]["input_images"]}/labels.csv", config["paths"]["input_images"]
)

# initialize the dataset - `snowPoleDataset()`
train_data = snowPoleDataset(
    training_samples,
    f"{config["paths"]["input_images"]}",
    aug=config["training"]["aug"],
)  ## we want all folders

valid_data = snowPoleDataset(
    valid_samples, f"{config["paths"]["input_images"]}", aug=False
)  # we always want the transform to be the normal transform

# prepare data loaders
train_loader = DataLoader(
    train_data, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    valid_data,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=0,
)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

if config["training"]["show_dataset_plot"]:
    utils.dataset_keypoints_plot(train_data)
    utils.dataset_keypoints_plot(valid_data)
