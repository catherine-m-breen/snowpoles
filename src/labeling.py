'''
written by Catherine M. Breen 
cbreen@uw.edu 

Use of our keypoint detection model currently requires ~10 images per camera. We provide a labeling script below that when pointed 
at a camera directory (i.e., data > cam1 or data > cam2, etc), walks the user through labeling every 10th image and saves as labels.csv in a specified direrctory. 

We estimate it will take about 5 imgs/min or about 300 imgs per hour. 

x1,y1 = top 
x2,y2 = bottom

The labels.csv file can then be directly pointed at train.py for fine-tuning. The user can then run predict.py to extract the snow depth.

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
    parser.add_argument('--datapath', help='Path to image dir', default = '/Volumes/CatBreen/Okanagan_Timelapse_Photos/CUB-U-01')
    parser.add_argument('--savedir', help='Path to save csv', default = '/Users/catherinebreen/Documents/Chapter1/WRRsubmission/data/conversions') 
    parser.add_argument('--pole_length', help='Length of pole in cm', default = '304.8')
    args = parser.parse_args()
        
    dir = glob.glob(f"{args.datapath}/**/*") #/*") ## path to data directory 
    dir = sorted(dir)

    filename = []
    PixelLengths = []
    topX, topY, bottomX, bottomY = [],[],[],[]
    creationTimes = []
    conversions = []

    pole_length = np.fromstring(args.pole_length)

    ### loop to label every 10th photo!
    for i, file in tqdm.tqdm(enumerate(dir)): 
       if i % 10 == 0: 
            img = cv2.imread(file)
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
                conversion = pole_length/PixelLength
            else: pass
            conversions.append(conversion)
            filename.append(file.split('/')[-1])
            creationTime = os.path.getctime(file)
            dt_c = datetime.datetime.fromtimestamp(creationTime)
            creationTimes.append(dt_c)

    # avg_conversion = np.average(conversions)
    # std_conversion = np.std(conversions)  
    df = pd.DataFrame({'filename':filename, 'datetime':creationTimes, 'x1':topX,'y1':topY, 'x2':bottomX, 'y2':bottomY, 'PixelLengths':PixelLengths, 
                       'conversions': conversions}) # 'mean': avg_conversion) # 'std': std_conversion})
    df.to_csv(f'{args.savedir}/labels.csv') # top10_conversion.csv')

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python snowpole_annotations.py --filepath '[insert filepath here]' ").
    main()