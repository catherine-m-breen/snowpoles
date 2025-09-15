## load in the old csv


## load up all the images 

## look up the image of interest 

## open that image and get the new ginput 



import cv2
import matplotlib.pyplot as plt
import glob
import argparse
import tqdm
import math
import pandas as pd
import os
import datetime
import numpy as np
from pathlib import Path
import tomli as tomllib
import IPython

def main():

    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description="Manually label images for training")
    parser.add_argument("--path", help="directory where images are located")
    parser.add_argument(
        "--datapath", help="(deprecated) directory where images are located"
    )
    parser.add_argument(
        "--image", help="label every N images"
    )
    parser.add_argument(
        "--no_confirm", required=False, help="skip confirmation", action="store_true"
    )
    args = parser.parse_args()
    args.path = args.datapath

    # Get arguments from config file if they weren't specified
    with open("config.toml", "rb") as configfile:
        config = tomllib.load(configfile)
    if not args.path:
        args.path = config["paths"]["input_images"]
    # if not args.pole_length:
    #     args.pole_length = config["labeling"]["pole_length"]
    if not args.subset_to_label:
        args.subset_to_label = config["labeling"]["subset_to_label"]

    # Confirmation
    if not args.no_confirm:
        print(
            "\n\n# The following options were specified in config.toml or as arguments:\n"
        )
        if (args.path.startswith("/")):
            print(
                "Directory where images are located:\n"
                + str(args.path)
                + "\n"
            )
        else:
            print(
                "Directory where images are located:\n"
                + os.getcwd()
                + "/"
                + str(args.path)
                + "\n"
            )
        #print("Pole length:\n" + str(args.pole_length) + "cm")
        print("\nImages to label:\nEvery", str(args.subset_to_label), "images")
        confirmation = str(input("\n\nIs this OK? (y/n) "))
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

    # dir = glob.glob(f"{args.path}/**/*")  # /*") ## path to data directory
    dir = list(
        Path(args.path).rglob("*.JPG")
    )  # Recursively lists all files and directories
    dir = sorted(dir)

    ## labeling data
    cameraids = []
    filename = []
    PixelLengths = []
    topX, topY, bottomX, bottomY = [], [], [], []
    creationTimes = []
    snowdepths = []

    ## customized data
    #pole_length = np.float64(args.pole_length)
    subset_to_label = np.int16(args.subset_to_label)

    ## load labels.csv
    write_headers_line = False
    try:
        with open(f"{args.path}/labels.csv", "r") as labels2_csv:
            lines = labels2_csv.readlines()
            with open(f"{args.path}/labels.csv", "w") as labels2_csv_write:
                for line in lines:
                    if line != "\n":
                        labels2_csv_write.write(line)
        with open(f"{args.path}/labels.csv", "r") as labels2_csv:
            if not labels2_csv.readline().startswith('"filename"'):
                write_headers_line = True
            else:
                for line in labels2_csv:
                    splitline = line.split(",")
                    cameraids.append(splitline[0])
                    filename.append(splitline[1])
                    creationTimes.append(splitline[2])
                    topX.append(splitline[3])
                    topY.append(splitline[4])
                    bottomX.append(splitline[5])
                    bottomY.append(splitline[6])
                    PixelLengths.append(splitline[7])
                    snowdepths.append(splitline[8].strip("\n"))

    except FileNotFoundError:
        write_headers_line = True
    if write_headers_line:
        print("labels.csv is corrupted or does not exist, creating...")
        with open(f"{args.path}/labels.csv", "w") as labels2_csv:
            labels2_csv.write(
                '"filename","datetime","x1","y1","x2","y2","PixelLengths","SnowDepths"'
            )

    if not os.path.exists(f"{args.path}/pole_metadata.csv"):
        ######## for pole_metdata #######
        processed_cameras = set()  # Track which cameras we've already processed
        meta_cameraids = []
        full_pole_length_pxs =[]
        pole_length_cms = []
        conversions = []
        heights = []
        widths = []

        for j, file in tqdm.tqdm(enumerate(dir)):
            # Skip if we've already processed this camera
            cameraID = Path(file).parent.name
            if cameraID in processed_cameras:
                continue
            processed_cameras.add(cameraID)
            img = cv2.imread(str(file))
            width, height, channel = img.shape
                ## assumes the cameras are stored in folder with their camera name
            figure = plt.figure(figsize=(20, 10), num=Path(file).name)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("label top and then bottom of 10cm section", fontweight="bold")
            top_10, bottom_10 = plt.ginput(2)
            plt.close()
            figure = plt.figure(figsize=(20, 10), num=Path(file).name)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("label top and then bottom of full pole", fontweight="bold")
            top, bottom = plt.ginput(2)
            plt.close()
            full_pole_length_px = math.dist((top), (bottom))
            full_pole_length_cm = (10 / math.dist((top_10), (bottom_10))) *  math.dist((top), (bottom))
            full_pole_length_pxs.append(full_pole_length_px)
            pole_length_cms.append(full_pole_length_cm), meta_cameraids.append(cameraID)

            conversion = full_pole_length_cm / full_pole_length_px 
            conversions.append(conversion)
            width, height, channel = img.shape
            heights.append(height), widths.append(width)
        
        pole_length_cm_lookup = dict(zip(meta_cameraids, pole_length_cms))
        conversion_lookup = dict(zip(meta_cameraids, conversions))

        metadata = pd.DataFrame(
            {
                "camera_id": pd.unique(meta_cameraids),
                "first_pole_length_px": full_pole_length_pxs,
                "pole_length_cm": pole_length_cms,
                "pixel_cm_conversion": conversions,
                "width": widths,
                "height": heights,
            }
        )
        metadata.to_csv(f"{args.path}/pole_metadata.csv", index=False)
    else: 
        metadata = pd.read_csv(f"{args.path}/pole_metadata.csv")
        pole_length_cm_lookup = dict(zip(metadata['camera_id'], metadata['pole_length_cm']))
        conversion_lookup = dict(zip(metadata['camera_id'], metadata['pixel_cm_conversion']))



    ### loop to label every nth photo!
    i = 0
    prev_cameraID = ""
    for j, file in tqdm.tqdm(enumerate(dir)):
        cameraID = Path(file).parent.name
        # whether to start counter over
        #i = i if len(cameraids) == 1 or cameraID == cameraids[-2] else 0
        if j == 0 or cameraID != Path(dir[j-1]).parent.name:
            i = 0

        if Path(file).name in filename:
            print(" ", Path(file).name, "has been labeled before, using stored data.")

        if i % subset_to_label == 0 and (not Path(file).name in filename):
            cameraids.append(cameraID)
            print(" ", Path(file).name)
            img = cv2.imread(str(file))
            width, height, channel = img.shape
            ## assumes the cameras are stored in folder with their camera name
            figure = plt.figure(figsize=(20, 10), num=Path(file).name)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("label top and then bottom", fontweight="bold")
            top, bottom = plt.ginput(2)
            topX.append(top[0]), topY.append(top[1])
            bottomX.append(bottom[0]), bottomY.append(bottom[1])
            plt.close()

            PixelLength = math.dist(top, bottom)
            PixelLengths.append(PixelLength)

            filename.append(Path(file).name)
            creationTime = os.path.getmtime(file)
            dt_c = datetime.datetime.fromtimestamp(creationTime)
            formatted_datetime = dt_c.strftime("%m/%d/%Y %H:%M")
            creationTimes.append(formatted_datetime)

            # ## snowdepth ##
            snowdepth = pole_length_cm_lookup[cameraID] - (PixelLength * conversion_lookup[cameraID])
            snowdepths.append(snowdepth)

            ## save data to labels.csv
            nextline = f"\n{cameraID},{Path(file).name},{formatted_datetime},{top[0]},{top[1]},{bottom[0]},{bottom[1]},{PixelLength},{snowdepth}"
            with open(f"{args.path}/labels.csv", "a") as labels2_csv:
                labels2_csv.write(nextline)

        i += 1

              ## snowdepth ##

    ## simplified table for snow depth conversion later on
    df = pd.DataFrame(
        {
            "cameraID":cameraids,
            "filename": filename,
            "datetime": creationTimes,
            "x1": topX,
            "y1": topY,
            "x2": bottomX,
            "y2": bottomY,
            "PixelLengths": PixelLengths,
            "SnowDepths":snowdepths,
        }
    )

    df.to_csv(f"{args.path}/labels.csv", index=False)

if __name__ == "__main__":
    main()
