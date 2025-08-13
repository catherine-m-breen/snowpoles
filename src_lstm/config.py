import torch
import os
from model_download import download_models
import tomli as tomllib
import IPython

# Read config.toml
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)

# Original settings
ROOT_PATH = config['paths']['input_images']
OUTPUT_PATH = config['paths']['models_output']
BATCH_SIZE = config['training']['batch_size']
LR = config['training']['lr']
EPOCHS = config['training']['epochs']
DEVICE = config['training']['device']
SHOW_DATASET_PLOT = config['training']['show_dataset_plot']
AUG = config['training']['aug']
FT_PATH = config['paths']['trainee_model']

# LSTM-specific settings
USE_LSTM = config['training']['use_lstm']
SEQUENCE_LENGTH = config['training']['sequence_length']
PROGRESSIVE_TRAINING = config['training']['progressive_training']
LSTM_HIDDEN_SIZE = config['training']['lstm_hidden_size']
LSTM_NUM_LAYERS = config['training']['lstm_num_layers']
LSTM_LR = config['training']['lstm_lr']
CNN_LR = config['training']['cnn_lr']

# Progressive training settings
STAGE1_EPOCHS = config['training']['stage1_epochs']
UNFREEZE_LAYERS = config['training']['unfreeze_layers']

# Existing settings
metadata = f"{ROOT_PATH}/pole_metadata.csv"
labels = f"{ROOT_PATH}/labels.csv"
predictions_output = config['paths']['images_output']
keypointColumns = ['x1', 'y1', 'x2', 'y2'] ## update

if not os.path.exists(FT_PATH):
    download_models("/".join(FT_PATH.split("/")[:-1]), FT_PATH.split("/")[-1])

# Helper function to get model configuration
def get_model_config():
    """Return model configuration dictionary"""
    return {
        'use_lstm': USE_LSTM,
        'sequence_length': SEQUENCE_LENGTH,
        'progressive_training': PROGRESSIVE_TRAINING,
        'lstm_hidden_size': LSTM_HIDDEN_SIZE,
        'lstm_num_layers': LSTM_NUM_LAYERS,
        'pretrained_cnn_path': FT_PATH if USE_LSTM else None,
        'stage1_epochs': STAGE1_EPOCHS,
        'unfreeze_layers': UNFREEZE_LAYERS
    }

def get_optimizer_config():
    """Return optimizer configuration for different training stages"""
    return {
        'base_lr': LR,
        'lstm_lr': LSTM_LR,
        'cnn_lr': CNN_LR,
        'progressive_training': PROGRESSIVE_TRAINING
    }