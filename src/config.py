import torch
import os

ROOT_PATH = '/datadrive/vmData/SNEX20_TLI_resized_clean'
OUTPUT_PATH = '/datadrive/vmData/snow_poles_outputs_resized_LRe4_BS64_E10000_clean_SNEX_IN_wOK_8020'
snowfreetbl_path = '/datadrive/vmData/snowfree_table.csv'
manual_labels_path = '/datadrive/vmData/manuallylabeled_CUBM02corr.csv' 
datetime_info = '/datadrive/vmData/labeledImgs_datetime_info.csv' 
native_res_path = '/datadrive/vmData/nativeRes.csv'

# learning parameters
BATCH_SIZE = 64 
LR = 0.0001
EPOCHS = 10000 #100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/test split
TEST_SPLIT = 0.2  ## could update for the cameras that we want to hold out as validation
# show dataset keypoint plot
SHOW_DATASET_PLOT = False
AUG = False 

keypointColumns = ['x1', 'y1', 'x2', 'y2'] ## update

# Fine-tuning set-up
FINETUNE = False 
FT_PATH = '/datadrive/vmData/snow_poles_outputs_resized_LRe4_BS64_E100_clean_SNEX_IN' ## model that you want to fine tune
FT_sample = 10
FT_IMG_PATH = '/datadrive/vmData/WAsubset_every10'
