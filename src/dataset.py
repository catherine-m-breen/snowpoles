'''
To do:
put in keypoint columns in the config file and update here

Updates: 
- switched train/test from random to 16 cameras : 4 cameras (OOD testing)
- specified columns in keypoints because we have extra columns in our df
- Specifed the __getitem__ function to look in nested folders of cameraIDs
    rather than training and testing
- specified the cameras for validation, 2 from each side, split from in and out of canopy
- hardcoded the training files path
- data aug docs: https://albumentations.ai/docs/getting_started/keypoints_augmentation/ 
- data aug docs cont. : https://albumentations.ai/docs/api_reference/augmentations/transforms/

'''

import torch
import cv2
import pandas as pd
import numpy as np
import config
#import config_cpu as config ## for cpu training
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

## to get a systematic sample: (already sorted)
# def sort_within_camera_group(group):
#     return group.sort_values(by='Filename')

# Define a function to sample every third photo
def sample_every_x(group, x):
    indices = np.arange(len(group[1]))
    every_x = len(group[1])//x
    selected_indices = indices[2::every_x]  # Select every third index starting from index 2
    return group[1].iloc[selected_indices]
#####

##### re-write this for out of domain testing
def train_test_split(csv_path, path, split, aug):
    #IPython.embed()
    df_data = pd.read_csv(csv_path)
    print(f'all rows in df_data {len(df_data.index)}')

    snex_cams = ['E6A', 'E6B', 'E9A','E9E', 'E9F','W1A','W2A','W2B',
            'W5A','W6A','W6B','W6C','W8A','W8C','W9A','W9B','W9C','W9D','W9E','W9G']
    wa_cams = ['TWISP-U-01', 'TWISP-R-01', 'CUB-H-02', 'CUB-L-02', 'CUB-M-02',
        'CEDAR-H-01', 'CEDAR-L-01', 'CEDAR-M-01','CUB-H-01','CUB-M-01','CUB-U-01', 'BUNKHOUSE-01']
    
    ### original model ### 
    ########## EXP #1: CAN MODEL DETECT SNOW  
        # if domain == True: 
        #     print('testing IN DOMAIN')
        #     training_samples = df_data.sample(frac=0.9, random_state=100) ## same shuffle everytime
        #     valid_samples = df_data[~df_data.index.isin(training_samples.index)]

        # else:
        #     print('testing OUT OF DOMAIN')
        #     ######### EXP #2: OUT OF DOMAIN TESTING ############
        #     val_cameras = ['E9E', 'W2E', 'CHE8', 'CHE9', 'TWISP-U-01']   ## would have to update this
        #     valid_samples = df_data[df_data['Camera'].isin(val_cameras)]  
        #     training_samples = df_data[~df_data['Camera'].isin(val_cameras)]

    ######### EXP #2 train just on SNEX cameras 
    #if snex == True:
    snex_data = df_data[df_data['Camera'].isin(snex_cams+wa_cams)] 

    training_samples = snex_data.sample(frac=0.8, random_state=100) ## same shuffle everytime
    valid_samples = snex_data[~snex_data.index.isin(training_samples.index)]
        # else:
        #     print('SNEX_WAOK cameras')
        #     valid_samples = valid_samples
        #     training_samples = training_samples
    
    wa_testdata = df_data[df_data['Camera'].isin(wa_cams)]  ## same no matter what
    co_testdata = valid_samples #df_data[df_data['Camera'].isin(snex_cams)] ## for fine-tuning

    if config.FINETUNE == True:
        print(f"FINETUNING MODEL n\ ")
        #IPython.embed()
        # stratsmp = glob.glob(f"{config.FT_IMG_PATH}/**/*")
        # stratsmp = [item.split('/')[-1] for item in stratsmp]
        # certain number every x from camera
        groups = wa_testdata.groupby('Camera')
        training_samples = pd.DataFrame()
        for group in groups: 
            y = sample_every_x(group, config.FT_sample)
            training_samples = pd.concat([training_samples, y])

        training_samples = training_samples
        valid_samples = wa_testdata[~wa_testdata['filename'].isin(training_samples['filename'])].sample(frac=0.1, random_state=100)  # just test on 10$ of WA data
        
        ## random x from each camera 
        # training_samples = wa_testdata.groupby('Camera').sample(config.FT_sample).reset_index() # X images per camera
        # valid_samples = wa_testdata[~wa_testdata.index.isin(training_samples.index)].sample(frac=0.2, random_state=100) ## could also just make this wa_testdata (all data)
        
        ## random sample
        # df_data = wa_testdata.sample(config.FT_sample).reset_index() # random sample
        
        # code before mar 21 2024
        # df_data = wa_testdata[wa_testdata['filename'].isin(stratsmp)].reset_index()
        # training_samples = df_data.sample(frac=0.9, random_state=100) ## same shuffle everytime
        # valid_samples = df_data[~df_data.index.isin(training_samples.index)]
        ######
        # valid_samples = wa_testdata.sample(frac=0.1, random_state=100)
        if not os.path.exists(f"{config.OUTPUT_PATH}"):
            os.makedirs(f"{config.OUTPUT_PATH}", exist_ok=True)
        training_samples.to_csv(f"{config.OUTPUT_PATH}/FT_training_samples.csv")
        valid_samples.to_csv(f"{config.OUTPUT_PATH}/FT_valid_samples.csv")

    ##### only images that exist
    all_images = glob.glob(path + ('/**/*.JPG'))
    filenames = [item.split('/')[-1] for item in all_images]
    valid_samples = valid_samples[valid_samples['filename'].isin(filenames)].reset_index()
    training_samples = training_samples[training_samples['filename'].isin(filenames)].reset_index()
    wa_testdata = wa_testdata[wa_testdata['filename'].isin(filenames)].reset_index()
    co_testdata = co_testdata[co_testdata['filename'].isin(filenames)].reset_index()
    
    if not os.path.exists(f"{config.OUTPUT_PATH}"):
            os.makedirs(f"{config.OUTPUT_PATH}", exist_ok=True)
    training_samples.to_csv(f"{config.OUTPUT_PATH}/training_samples.csv")
    valid_samples.to_csv(f"{config.OUTPUT_PATH}/valid_samples.csv")

    print(f'# of examples we will now train on {len(training_samples)}, val on {len(valid_samples)}')
    print('LATEST CODE CHECK 2')
    return training_samples, valid_samples, wa_testdata, co_testdata


class snowPoleDataset(Dataset):

    def __init__(self, samples, path, aug): # split='train'):
        self.data = samples
        self.path = path
        self.resize = 224
        # self.transform = Compose([              # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
        #     Resize(([224, 224])), 
        #            # For now, we just resize the images to the same dimensions...
        #     ToTensor()                          # ...and convert them to torch.Tensor.
        # ])
        if aug == False: 
            self.transform = A.Compose([
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy'))
        else: 
            self.transform = A.Compose([
                A.ToFloat(max_value=1.0),
                A.CropAndPad(px=75, p =1.0), ## final model is 50 pixels
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=20, p=0.5),
                #A.OneOf([
                 #   A.Affine(translate_px = (-3, 3),p=0.5), ### will throw off algorithm 
                  #  A.Affine(scale = (0.5, 1.0), p =0.5),
                   # A.Affine(translate_percent = (-0.15,0.15), p =0.5)], p =0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.ToGray(p=0.5)], p = 0.5),
                #A.FromFloat(max_value=1.0),
                A.Resize(224, 224),
                ], keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.data)

    def __filename__(self, index):
        #print('test')
        filename = self.data.iloc[index]['filename']
        return filename
    
    def __getitem__(self, index):
        cameraID = self.data.iloc[index]['filename'].split('_')[0] ## need this to get the right folder
        filename = self.data.iloc[index]['filename']
        #IPython.embed()
        
        image = cv2.imread(f"{self.path}/{cameraID}/{self.data.iloc[index]['filename']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        #IPython.embed()
        # again reshape to add grayscale channel format
        image = image / 255.0
        # transpose for getting the channel size to index 0
        #image = np.transpose(image, (2, 0, 1))

        ### PIL library
        #img = Image.open(f"{self.path}/{cameraID}/{self.data.iloc[index]['filename']}").convert('RGB')  
        #orig_h, orig_w = img.size

        # get the keypoints
        keypoints = self.data.iloc[index][1:][['x1','y1','x2','y2']]  #[3:7]  ### change to x1 y1 x2 y2
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)
        # rescale keypoints according to image resize
        #IPython.embed()
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]

        #=if config.RANDOM_ROTATION == True: 
         #   image = T.ColorJitter(brightness=.5, hue=.3)
        #utils.vis_keypoints(image, keypoints)
        #IPython.embed()
        transformed = self.transform(image=image, keypoints=keypoints)
        img_transformed = transformed['image']
        keypoints = transformed['keypoints']
        #IPython.embed()
        #utils.vis_keypoints(transformed['image'], transformed['keypoints'])
        image = np.transpose(img_transformed, (2, 0, 1))
        #IPython.embed()
        if len(keypoints) != 2:
            #IPython.embed()
            utils.vis_keypoints(transformed['image'], transformed['keypoints'])

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
            'filename': filename
        }

# get the training and validation data samples
# we also added a test set for wa cameras that we will adjust after some fine-tuning
training_samples, valid_samples, wa_testdata, co_testdata = train_test_split(f"{config.ROOT_PATH}/snowPoles_labels_clean_jul23upd.csv", f"{config.ROOT_PATH}", 
                                                   config.TEST_SPLIT, config.AUG)

# initialize the dataset - `snowPoleDataset()`
train_data = snowPoleDataset(training_samples, 
                                 f"{config.ROOT_PATH}", aug = config.AUG)  ## we want all folders
#IPython.embed()
valid_data = snowPoleDataset(valid_samples, 
                                 f"{config.ROOT_PATH}", aug = False) # we always want the transform to be the normal transform

wa_data = snowPoleDataset(wa_testdata, 
                            f"{config.ROOT_PATH}", aug = False) # this will be some assortment of the WA A & B cameras

co_data = snowPoleDataset(co_testdata, 
                            f"{config.ROOT_PATH}", aug = False) # this will be some assortment of the WA A & B cameras

# prepare data loaders
train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True, num_workers = 0)
valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False, num_workers = 0) 

# test_loader = DataLoader(test_data, 
#                           batch_size=config.BATCH_SIZE, 
#                           shuffle=False, num_workers = 0) 

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")
print(f"Test WA sample instances: {len(wa_data)}")

# whether to show dataset keypoint plots
if config.SHOW_DATASET_PLOT:
    utils.dataset_keypoints_plot(train_data)
    utils.dataset_keypoints_plot(valid_data)




