'''
load model results and identify snow depth
'''

import torch
import numpy as np
import cv2
import albumentations  ## may need to do pip install
import config
from model import snowPoleResNet50
import argparse
import glob
import IPython
import utils
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Predict top and bottom coordinates.')
    parser.add_argument('--model_path', required=False, help = 'Path to model', default = 'NULL')
    parser.add_argument('--dir_path', required=False, help='Path to camera image directory', default = '/example_data')
    parser.add_argument('--folder_path', required=False, help='Path to camera image folder', default = "/example_data/cam1")
    parser.add_argument('--output_path', required=True, help='Path to output folder', default = "/example_data")
    args = parser.parse_args()


    import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import IPython
from scipy.spatial import distance

IPython.embed()

# data = pd.read_csv('/Users/catherinebreen/Dropbox/Chapter1/dendrite_outputs/OUT/snow_poles_outputs_resized_LRe4_BS64_clean_wWA_OUT_earlystop/results.csv')
data = pd.read_csv('/Users/catherinebreen/Documents/Chapter1/dendrite_outputs/OUT/snow_poles_outputs_resized_LRe4_BS64_clean_wWAOK_OUT/eval/results.csv')
snwfreetbl = pd.read_csv('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/snowfree_table.csv') # updated, make sure these tables have the meta information for the sites of interest

nativeRes_imgs = glob.glob("/Users/catherinebreen/Documents/Chapter1/WRRsubmission/resolution_info/*")
camIDs = []
nativeRes = []

## turn into dictionary 
for img in nativeRes_imgs:
    camID = img.split('/')[-1].split('_')[0]
    image = cv2.imread(img)
    orig_h, orig_w, channel = image.shape
    camIDs.append(camID)
    nativeRes.append([orig_h, orig_w])

resDic = dict(zip(camIDs, nativeRes))

# dates = pd.read_csv('/Users/catherinebreen/Documents/Chapter1/WRRsubmission/labeledImgs_datetime_info.csv') # updated, make sure these tables have the meta information for the sites of interest
# dates.rename(columns = {'filenames' : 'filename'}, inplace = True)

# data = data.merge(dates, on='filename', how='left') ## make sure that the datetime format is right

files = []
# che_ok_dates = []
cam = []
length_cm = []
sd = []

for filename in data['filename']:
    try: 
        camera = filename.split('_')[0]
        res = resDic[camera]
        conversion = snwfreetbl.loc[snwfreetbl['camera'] == camera, 'conversion'].iloc[0]
        snwfreestake = snwfreetbl.loc[snwfreetbl['camera'] == camera, 'snow_free_cm'].iloc[0]
        #IPython.embed()
        ## need to scale back up 
        x1 = data.loc[data['filename'] == filename, 'x1_pred'].iloc[0] * (res[1] / 224)
        y1 = data.loc[data['filename'] == filename, 'y1s_pred'].iloc[0] * (res[0] / 224)
        x2 = data.loc[data['filename'] == filename, 'x2_pred'].iloc[0] * (res[1] / 224)
        y2 = data.loc[data['filename'] == filename, 'y2_pred'].iloc[0] * (res[0] / 224)

        pixelLengths = distance.euclidean([x1,y1],[x2,y2])
        cmLengths = pixelLengths * float(conversion)
        snowdepth = snwfreestake - cmLengths

        #if snowdepth < 0: snowdepth = 0 # so that we reduce the noise

        files.append(filename), 
        cam.append(camera), length_cm.append(cmLengths), sd.append(snowdepth)
    except: pass

df = pd.DataFrame({'camera': cam, 'filename': files, 'cmLengths': length_cm,'snowDepth':sd})
df.to_csv('/Users/catherinebreen/Dropbox/Chapter1/dendrite_outputs/OUT/snow_poles_outputs_resized_LRe4_BS64_clean_wWAOK_OUT/results_cm.csv')

IPython.embed()
    #results = eval(outputs)

if __name__ == '__main__':
    main()



