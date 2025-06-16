import torch
import os
from model_download import download_models

"""
!! Update ROOT_PATH before running !!
"""

ROOT_PATH = "../Snow Station Photos (BigW)/Current/2023-2024"  ## Folder where images and CSVs are stored
OUTPUT_PATH = "./output1"  ## the folder where you want to store your custom model

# learning parameters
BATCH_SIZE = 64
LR = 0.0001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train/test split
TEST_SPLIT = 0.2

# Fine-tuning set-up
FINETUNE = True

# show dataset keypoint plot
SHOW_DATASET_PLOT = False
AUG = True

# Path to model
FT_PATH = "./models/CO_and_WA_model.pth"


metadata = f"{ROOT_PATH}/pole_metadata.csv"
labels = f"{ROOT_PATH}/labels.csv"

keypointColumns = ["x1", "y1", "x2", "y2"]  ## update

if not os.path.exists(FT_PATH):
    download_models(FT_PATH.split("/")[:-1], FT_PATH.split("/")[-1])
