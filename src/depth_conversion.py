'''
written by Catherine Breen 
June 2024

If after the predictions you want to predict snow depth again
such as if you have improved metadata, you can run this script by itself on the predictions and the metadata

example command line to run:

python src/depth_conversion.py

'''


import numpy as np
from model import snowPoleResNet50
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance

def main():
    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description='Predict top and bottom coordinates.')
    parser.add_argument('--predictions_path', required=True, help='Path to camera image directory', default = '/predictions/results.csv')
    parser.add_argument('--metadata', required=False, help='Path to pole metadata', default = "/example_data/pole_metadata.csv")
    parser.add_argument('')
    args = parser.parse_args()

    predictions = pd.read_csv(f'{args.predictions_path}')
    metadata = pd.read_csv(f"{args.metadata}")

    files = []
    cameras =[]
    snow_depths = []

    for filename in predictions['filename']:
        try: 
            camera = filename.split('/')[-1]
        
            full_length_pole_cm = metadata.loc[metadata['camera_id'] == camera, 'pole_length'].iloc[0]
            pixel_cm_conversion = metadata.loc[metadata['camera_id'] == camera, 'pixel_cm_conversion'].iloc[0]
            #IPython.embed()
            ## need to scale back up 
            x1 = predictions.loc[predictions['filename'] == filename, 'x1_pred'].iloc[0] 
            y1 = predictions.loc[predictions['filename'] == filename, 'y1_pred'].iloc[0] 
            x2 = predictions.loc[predictions['filename'] == filename, 'x2_pred'].iloc[0] 
            y2 = predictions.loc[predictions['filename'] == filename, 'y2_pred'].iloc[0] 

            total_length_pixel = distance.euclidean([x1,y1],[x2,y2])
            snow_depth = full_length_pole_cm - (pixel_cm_conversion * total_length_pixel)
            
            files.append(filename)
            cameras.append(camera)
            snow_depths.append(snow_depth)

        except: pass

    df = pd.DataFrame({'camera_id': cameras, 'filename': files, 'snowdepth':snow_depths})
    df.to_csv(f'{args.predictions_path}/results_wsnowdepthcm.csv')

if __name__ == '__main__':
    main()



