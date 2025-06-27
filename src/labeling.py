"""
written by Catherine M. Breen 
cbreen@uw.edu 

Use of our keypoint detection model currently requires ~10 images per camera. We provide a labeling script below that when pointed 
at a camera directory (i.e., data > cam1 or data > cam2, etc), walks the user through labeling every 10th image and saves as labels.csv in a specified direrctory. 

We estimate it will take about 5 imgs/min or about 300 imgs per hour. 

x1,y1 = top 
x2,y2 = bottom

The labels.csv file can then be directly pointed at train.py for fine-tuning. The user can then run predict.py to extract the snow depth.

example run 

python src/labeling.py --datapath "/path/to/nontrained/data" --pole_length "304.8" --subset_to_label "2"

"""

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
import tomllib

# Comment out this line to disable dark mode
plt.style.use("./themes/dark.mplstyle")


def main():

    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description="Manually label images for training")
    parser.add_argument("--path", help="directory where images are located")
    parser.add_argument(
        "--datapath", help="(deprecated) directory where images are located"
    )
    parser.add_argument("--pole_length", help="length of pole in cm")
    parser.add_argument("--subset_to_label", help="label every N images")
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
    if not args.pole_length:
        args.pole_length = config["labeling"]["pole_length"]
    if not args.subset_to_label:
        args.subset_to_label = config["labeling"]["subset_to_label"]

    # Confirmation
    if not args.no_confirm:
        print(
            "\n\n# The following options were specified in config.toml or as arguments:\n"
        )
        if args.path.startswith("/") or args.path[1] == ":":
            print("Directory where images are located:\n" + str(args.path) + "\n")
        else:
            print(
                "Directory where images are located:\n"
                + os.getcwd()
                + "/"
                + str(args.path)
                + "\n"
            )
        print("Pole length:\n" + str(args.pole_length) + "cm")
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

    label_photos(args.path, args.pole_length, args.subset_to_label)

def label_photos(path, pole_length, subset_to_label):
    dir = list(Path(path).rglob("*.JPG"))
    dir = sorted(dir)

    ## labeling data
    filename = []
    PixelLengths = []
    topX, topY, bottomX, bottomY = [], [], [], []
    creationTimes = []

    ## customized data
    pole_length_f64 = np.float64(pole_length)
    subset_to_label_i16 = np.int16(subset_to_label)

    ## some metadata data
    cameraIDs = []
    pole_lengths = []  ## tracks pole length
    first_pole_pixel_length = []
    conversions = []
    widths, heights = [], []

    ## load labels.csv
    write_headers_line = False
    try:
        with open(f"{path}/labels.csv", "r") as labels2_csv:
            lines = labels2_csv.readlines()
            with open(f"{path}/labels.csv", "w") as labels2_csv_write:
                for line in lines:
                    if line != "\n":
                        labels2_csv_write.write(line)
        with open(f"{path}/labels.csv", "r") as labels2_csv:
            if not labels2_csv.readline().startswith('"filename"'):
                write_headers_line = True
            else:
                for line in labels2_csv:
                    splitline = line.split(",")
                    filename.append(splitline[0])
                    creationTimes.append(splitline[1])
                    topX.append(splitline[2])
                    topY.append(splitline[3])
                    bottomX.append(splitline[4])
                    bottomY.append(splitline[5])
                    PixelLengths.append(splitline[6].strip("\n"))
    except FileNotFoundError:
        write_headers_line = True
    if write_headers_line:
        print("labels.csv is corrupted or does not exist, creating...")
        with open(f"{path}/labels.csv", "w") as labels2_csv:
            labels2_csv.write(
                '"filename","datetime","x1","y1","x2","y2","PixelLengths"'
            )

    ### loop to label every nth photo!
    i = 0
    prev_cameraID = ""
    for j, file in tqdm.tqdm(enumerate(dir)):
        cameraID = Path(file).parent.name
        cameraIDs.append(cameraID)

        ##whether to start counter over
        i = i if len(cameraIDs) == 1 or cameraID == cameraIDs[-2] else 0

        if Path(file).name in filename:
            print(" ", Path(file).name, "has been labeled before, using stored data.")

        if i % subset_to_label_i16 == 0 and (not Path(file).name in filename):
            print(" ", Path(file).name)
            img = cv2.imread(file)
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

            ## save data to labels.csv
            nextline = f"\n{Path(file).name},{os.path.getctime(file)},{top[0]},{top[1]},{bottom[0]},{bottom[1]},{PixelLength}"
            with open(f"{path}/labels.csv", "a") as labels2_csv:
                labels2_csv.write(nextline)

            filename.append(Path(file).name)
            creationTime = os.path.getctime(file)
            dt_c = datetime.datetime.fromtimestamp(creationTime)
            creationTimes.append(dt_c)

        if not len(prev_cameraID) or cameraID != prev_cameraID:
            prev_cameraID = cameraID
            mj = int(j / subset_to_label)
            PixelLength = math.dist(
                (float(topX[mj]), float(topY[mj])),
                (float(bottomX[mj]), float(bottomY[mj])),
            )
            ## with the first photo, we will get some metadata
            conversion = pole_length / PixelLength
            ## and get metadata
            first_pole_pixel_length.append(PixelLength)
            conversions.append(conversion)
            pole_lengths.append(pole_length)
            img = cv2.imread(file)
            width, height, channel = img.shape
            heights.append(height), widths.append(width)

        i += 1

    ## simplified table for snow depth conversion later on
    metadata = pd.DataFrame(
        {
            "camera_id": pd.unique(cameraIDs),
            "pole_length_cm": pole_lengths,
            "pole_length_px": first_pole_pixel_length,
            "pixel_cm_conversion": conversions,
            "width": widths,
            "height": heights,
        }
    )

    metadata.to_csv(f"{path}/pole_metadata.csv")


if __name__ == "__main__":
    main()
