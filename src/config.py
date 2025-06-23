import torch
import os
from model_download import download_models
import tomllib

# Read config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)

ROOT_PATH = config["paths"]["input_images"]
OUTPUT_PATH = config["paths"]["models_output"]
BATCH_SIZE = config["training"]["batch_size"]
LR = config["training"]["lr"]
EPOCHS = config["training"]["epochs"]
DEVICE = config["training"]["device"]
SHOW_DATASET_PLOT = config["training"]["show_dataset_plot"]
AUG = config["training"]["aug"]
FT_PATH = config["paths"]["trainee_model"]

metadata = f"{ROOT_PATH}/pole_metadata.csv"
labels = f"{ROOT_PATH}/labels.csv"

keypointColumns = ['x1', 'y1', 'x2', 'y2'] ## update

if not os.path.exists(FT_PATH):
    download_models("/".join(FT_PATH.split("/")[:-1]), FT_PATH.split("/")[-1])
