import torch
import os

### THESE PATHS ARE FOR THE DEMO NONTRAINED DATA; PLEASE UPDATE WITH OWN PATHS"
### All in snowpoles_data
ROOT_PATH = './example_nontrained_data'
OUTPUT_PATH = './output1'  ## the folder where you want to store your custom model
metadata = './example_nontrained_data/pole_metadata.csv'
labels = './example_nontrained_data/labels.csv'

# learning parameters
BATCH_SIZE = 64 
LR = 0.0001
EPOCHS = 20 #1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
TEST_SPLIT = 0.2 
# show dataset keypoint plot
SHOW_DATASET_PLOT = False
AUG = True 

keypointColumns = ['x1', 'y1', 'x2', 'y2'] ## update

# Fine-tuning set-up
FINETUNE = True 
FT_PATH = './models/CO_and_WA_model.pth' 
