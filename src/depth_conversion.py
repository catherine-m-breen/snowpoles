'''
load model results and identify snow depth for AVERAGE USER GITHUB 
REPOSITORY

Catherine Breen
catherine.m.breen@gmail.com

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
from PIL import Image

def main():
    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description='Predict top and bottom coordinates.')
    parser.add_argument('--predictions_path', required=True, help='Path to camera image directory', default = '/predictions')
    parser.add_argument('--snwfreetbl_path,', required=True, help='Path to snow free table', default = '/snowfree_table.csv')
    parser.add_argument('--native_res_info', required=True, help='Path to example image folder', default = '/snowfree_table.csv')
    args = parser.parse_args()

    data = pd.read_csv(f'{args.predictions_path}/eval/results.csv')
    snwfreetbl = pd.read_csv(f'{args.snwfreetbl_path}/snowfree_table.csv') 
    nativeRes_imgs = glob.glob(f'{args.native_res_info}/resolution_info/*')

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

    files = []
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
    df.to_csv(f'{args.predictions_path}/results_cm.csv')

if __name__ == '__main__':
    main()



