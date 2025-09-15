
'''
python pole_metadata_redo.py --path '/Users/cmbreen/Documents/snow/alaska_dataset/all_images/BC_final_448'

python pole_metadata_redo.py --path '/Users/cmbreen/Documents/snow/alaska_dataset/all_images/snowfree_photos/CP_final_448'


'''

import pandas as pd
import tqdm
import cv2
import matplotlib.pyplot as plt
import glob
import argparse
import math
import os
import datetime
import numpy as np
from pathlib import Path
import tomli as tomllib
import IPython

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process pole measurement images and generate metadata')
    parser.add_argument('--path', '-p', 
                       type=str, 
                       required=True,
                       help='Path to the directory containing camera folders with image files')

    args = parser.parse_args()
    
    # Validate path exists
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist")
        return
    
    # Get all camera folders (subdirectories)
    camera_folders = [f for f in os.listdir(args.path) 
                     if os.path.isdir(os.path.join(args.path, f))]
    
    if not camera_folders:
        print(f"No camera folders found in '{args.path}'")
        return
    
    print(f"Found {len(camera_folders)} camera folders to process")

    if not os.path.exists(f"{args.path}/pole_metadata.csv"):
        ######## for pole_metadata #######
        meta_cameraids = []
        full_pole_length_pxs = []
        pole_length_cms = []
        conversions = []
        heights = []
        widths = []
        top10section = []

        for camera_folder in tqdm.tqdm(camera_folders, desc="Processing cameras"):
            camera_path = os.path.join(args.path, camera_folder)
            
            # Get all images in this camera folder, sorted by name
            image_files = sorted(glob.glob(os.path.join(camera_path, '*.JPG')))
            
            # Process images in order until we get a valid measurement
            valid_measurement = False
            for i, file in enumerate(image_files):
                print(f"  Trying image {i+1}/{len(image_files)}: {os.path.basename(file)}")
                
                img = cv2.imread(str(file))
                height, width, channel = img.shape
                
                # Display image for 10cm section measurement
                figure = plt.figure(figsize=(20, 10), num=f"{camera_folder} - 10cm section")
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"Camera: {camera_folder} - Label top and then bottom of 10cm section", fontweight="bold")
                top_10, bottom_10 = plt.ginput(2)
                plt.close()
                
                # Display image for full pole measurement
                figure = plt.figure(figsize=(20, 10), num=f"{camera_folder} - Full pole")
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"Camera: {camera_folder} - Label top and then bottom of full pole", fontweight="bold")
                top, bottom = plt.ginput(2)
                plt.close()
                
                # Calculate measurements
                full_pole_length_px = math.dist(top, bottom)
                
                print(f"    Top portion: {full_pole_length_px:.2f} pixels")
                
                # Check if measurement is valid (> 5 pixels)
                if full_pole_length_px > 5:
                    print(f"    ✓ Valid measurement found for {camera_folder}")
                    
                    # Calculate final measurements
                    ten_cm_length_px = math.dist(top_10, bottom_10)
                    conversion = 10 / ten_cm_length_px ## ten_cm_length_px
                    full_pole_length_cm = conversion * full_pole_length_px
                    
                    
                    # Store the results
                    meta_cameraids.append(camera_folder)
                    full_pole_length_pxs.append(full_pole_length_px)
                    pole_length_cms.append(full_pole_length_cm)
                    top10section.append(ten_cm_length_px)
                    conversions.append(conversion)
                    heights.append(height)
                    widths.append(width)
                    
                    valid_measurement = True
                    print(f"top10section: {top10section} \n full pole length px : {full_pole_length_px} \n pole length cm: {full_pole_length_cm}")
                    break
                else:
                    print(f"    ✗ Measurement too small ({full_pole_length_px:.2f}px), trying next image...")
            
            if not valid_measurement:
                print(f"  ⚠️  No valid measurements found for camera: {camera_folder}")
        
        if meta_cameraids:
            # Create lookup dictionaries
            pole_length_cm_lookup = dict(zip(meta_cameraids, pole_length_cms))
            conversion_lookup = dict(zip(meta_cameraids, conversions))

            # Create metadata DataFrame
            metadata = pd.DataFrame(
                {
                    "camera_id": meta_cameraids,
                    "first_pole_length_px": full_pole_length_pxs,
                    "full_pole_length_cm": pole_length_cms,
                    "top10_section_px": top10section,
                    "pixel_cm_conversion": conversions,
                    "width": widths,
                    "height": heights,
                }
            )
            metadata.to_csv(f"{args.path}/pole_metadata3.csv", index=False)
            print(f"\n✓ Metadata saved to {args.path}/pole_metadata3.csv")
            print(f"Successfully processed {len(meta_cameraids)} cameras")
        else:
            print("\n⚠️  No valid measurements obtained from any camera")
    else:
        print(f"Metadata file already exists at {args.path}/pole_metadata.csv")

if __name__ == "__main__":
    main()