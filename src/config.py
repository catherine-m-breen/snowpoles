import torch
import os

### THESE PATHS ARE FOR THE DEMO DATA; PLEASE UPDATE WITH OWN PATHS"
### All in snowpoles_data
ROOT_PATH = '~/example_data'
OUTPUT_PATH = '~/output'
snowfreetbl_path = '~/snowpoles_data/demo_data/snowfree_table.csv'
manual_labels_path = '~/example_data/snowPoles_labels_clean_448res.csv' 
# datetime_info = '~/labeledImgs_datetime_info.csv' 
# native_res_path = '~/nativeRes.csv'

# learning parameters
BATCH_SIZE = 64 
LR = 0.0001
EPOCHS = 10000 #100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
TEST_SPLIT = 0.2 
# show dataset keypoint plot
SHOW_DATASET_PLOT = False
AUG = False 

keypointColumns = ['x1', 'y1', 'x2', 'y2'] ## update

# Fine-tuning set-up
FINETUNE = False 
FT_PATH = '~/model/model.pth' ## model that you want to fine tune
FT_sample = 10
FT_IMG_PATH = '~/example_data'
