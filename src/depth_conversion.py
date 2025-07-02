'''
written by Catherine Breen 
June 2024

If after the predictions you want to predict snow depth again
such as if you have improved metadata, you can run this script by itself on the predictions and the metadata

example command line to run:

python src/depth_conversion.py --predictions_path '/predictions/results.csv' --metadata 'example_nontrained_data/pole_metadata.csv'

'''


import numpy as np
import tomli as tomllib
import os
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
import IPython
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert pixel lengths into snow depth")
    parser.add_argument("--path", help="directory where images/metadata are located")
    parser.add_argument(
        "--output", required=False, help="directory in which to store marked images"
    )
    parser.add_argument(
        "--no_confirm", required=False, help="skip confirmation", action="store_true"
    )
    global args
    args = parser.parse_args()

    # Get arguments from config file if they weren't specified
    with open("config.toml", "rb") as configfile:
        config = tomllib.load(configfile)
    if not args.path:
        args.path = config["paths"]["input_images"]
    if not args.output:
        args.output = config["paths"]["images_output"]

    # Confirmation
    if not args.no_confirm:
        print(
            "\n\n# The following options were specified in config.toml or as arguments:\n"
        )
        if (args.path.startswith("/")):
            print(
                "Directory where images/metadata are located:\n"
                + str(args.path)
                + "\n"
            )
        else:
            print(
                "Directory where images/metadata are located:\n"
                + os.getcwd()
                + "/"
                + str(args.path)
                + "\n"
            )
        if (args.output.startswith("/")):
            print(
                "Directory where results with snow depth will be stored:\n"
                + str(args.output)
                + "\n"
            )
        else:
            print(
                "Directory where results with snow depth will be stored:\n"
                + os.getcwd()
                + "/"
                + str(args.output)
                + "\n"
            )

        confirmation = str(input("\nIs this OK? (y/n) "))
        if confirmation.lower() != "y":
            if confirmation.lower() == "n":
                print(
                    "\nEdit the config file, located at",
                    os.getcwd()
                    + "/config.toml, to your liking, or edit the command line arguments if they were specified, and then re-run this file.\n",
                )
            else:
                print("Invalid input.\n")
            quit()


    predictions = pd.read_csv(f'{args.output}/results.csv')
    metadata = pd.read_csv(f"{args.path}/pole_metadata.csv")

    files = []
    cameras =[]
    snow_depths = []

    for filename in tqdm(predictions['filename']):
        try: 
            camera = Path(predictions['filename'][0]).name.split('_')[0]
        
            full_length_pole_cm = metadata.loc[metadata['camera_id'] == camera, 'pole_length_cm'].iloc[0]
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
    df.to_csv(f'{args.output}/results_wsnowdepthcm.csv')

    print(f'saved at {args.output}/results_wsnowdepthcm.csv')

if __name__ == '__main__':
    main()



