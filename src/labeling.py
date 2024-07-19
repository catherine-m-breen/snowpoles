'''
written by Catherine M. Breen 
cbreen@uw.edu 

Use of our keypoint detection model currently requires ~10 images per camera. We provide a labeling script below that when pointed 
at a camera directory (i.e., data > cam1 or data > cam2, etc), walks the user through labeling every 10th image and saves as labels.csv in a specified direrctory. 

We estimate it will take about 5 imgs/min or about 300 imgs per hour. 

x1,y1 = top 
x2,y2 = bottom

The labels.csv file can then be directly pointed at train.py for fine-tuning. The user can then run predict.py to extract the snow depth.

example run 

python src/labeling.py --datapath 'example_nontrained_data' --pole_length '304.8' --subset_to_label '10'

'''

import cv2
import matplotlib.pyplot as plt 
import glob
import argparse
import tqdm
import math
import pandas as pd
import os
import datetime
import IPython
import numpy as np 

def main():

    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description='Label snowpole images')
    parser.add_argument('--datapath', help='Path to image dir')
    parser.add_argument('--pole_length', help='Length of pole in cm', default = '100') ### assumes poles are 1 m / 100 cm tall 
    parser.add_argument('--subset_to_label', help='# of images per camera to label', default = '10')
    args = parser.parse_args()
        
    dir = glob.glob(f"{args.datapath}/**/*") #/*") ## path to data directory 
    dir = sorted(dir)

    ## labeling data
    filename = []
    PixelLengths = []
    topX, topY, bottomX, bottomY = [],[],[],[]
    creationTimes = []

    ## customized data
    pole_length = np.fromstring(args.pole_length)
    subset_to_label = np.fromstring(args.pole_length)

    ## some metadata data
    cameraIDs= []
    pole_lengths = [] ## tracks pole length
    first_pole_pixel_length = []
    conversions = []
    widths, heights = [], []

    ### loop to label every nth photo!
    for i, file in tqdm.tqdm(enumerate(dir)): 
       if i % subset_to_label == 0: 
            img = cv2.imread(file)
            width, height, channel = img.shape
            ## assumes the cameras are stored in folder with their camera name 
            cameraID = file.split('/')[-1] 
            plt.figure(figsize = (20,10))
            plt.imshow(img)
            plt.title('label top and then bottom', fontweight = "bold")
            top, bottom = plt.ginput(2)
            topX.append(top[0]), topY.append(top[1])
            bottomX.append(bottom[0]), bottomY.append(bottom[1])
            plt.close()

            PixelLength = math.dist(top,bottom)
            PixelLengths.append(PixelLength)
            
            ## to get the pixel to centimeter conversion

            if i == 0:
                ## with the first photo, we will get some metadata
                conversion = pole_length/PixelLength
                ## and get metadata
                first_pole_pixel_length.append(PixelLength)
                conversions.append(conversion)
                pole_lengths.append(pole_length)
                heights.append(height), widths.append(width)

            else: pass
            filename.append(file.split('/')[-1])
            creationTime = os.path.getctime(file)
            dt_c = datetime.datetime.fromtimestamp(creationTime)
            creationTimes.append(dt_c)
            cameraIDs.append(cameraID)

    df = pd.DataFrame({'filename':filename, 'datetime':creationTimes, 'x1':topX,'y1':topY, 'x2':bottomX,
                        'y2':bottomY, 'PixelLengths':PixelLengths}) 
    
    ## simplified table for snow depth conversion later on
    metadata = pd.DataFrame({'camera_id':pd.unique(cameraIDs), 'first_pole_length_cm':pole_lengths,
                             'first_pole_length_px':(first_pole_pixel_length), 
                             'conversion':pd.unique(conversions),'width':widths,'height':heights})
    
    df.to_csv(f'{args.savedir}/labels.csv') 
    metadata.to_csv(f'{args.savedir}/metadata.csv')

if __name__ == '__main__':
    main()
