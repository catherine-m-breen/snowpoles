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
python src/labeling.py --datapath "/Users/cmbreen/Documents/FDLTCC/FF_2024" --subset_to_label "10"

python src/labeling.py --datapath "/Users/cmbreen/Documents/FDLTCC/summer_2025/FF_2024" --subset_to_label "10"



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
import tomli as tomllib
import IPython

def enable_scroll_zoom_and_pan(ax, base_scale=1.2):
    """Enables mouse-wheel zooming and right-click panning for a matplotlib axis"""
    pan_state = {'is_panning': False, 'start_x': None, 'start_y': None, 'start_xlim': None, 'start_ylim': None}

    def zoom(event):
        if event.inaxes != ax: return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata 
        ydata = event.ydata 
        
        if event.button == 'up':
            scale_factor = 1 / base_scale # zoom in
        elif event.button == 'down':
            scale_factor = base_scale     # zoom out
        else:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        ax.figure.canvas.draw_idle()

    def press(event):
        # Button 3 is the RIGHT mouse button
        if event.button == 3 and event.inaxes == ax:
            pan_state['is_panning'] = True
            pan_state['start_x'] = event.x
            pan_state['start_y'] = event.y
            pan_state['start_xlim'] = ax.get_xlim()
            pan_state['start_ylim'] = ax.get_ylim()

    def release(event):
        if event.button == 3:
            pan_state['is_panning'] = False

    def motion(event):
        if pan_state['is_panning'] and pan_state['start_x'] is not None:
            dx_pixels = event.x - pan_state['start_x']
            dy_pixels = event.y - pan_state['start_y']
            bbox = ax.get_window_extent()
            dx_data = dx_pixels * (pan_state['start_xlim'][1] - pan_state['start_xlim'][0]) / bbox.width
            dy_data = dy_pixels * (pan_state['start_ylim'][1] - pan_state['start_ylim'][0]) / bbox.height
            
            ax.set_xlim(pan_state['start_xlim'][0] - dx_data, pan_state['start_xlim'][1] - dx_data)
            ax.set_ylim(pan_state['start_ylim'][0] - dy_data, pan_state['start_ylim'][1] - dy_data)
            ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('scroll_event', zoom)
    ax.figure.canvas.mpl_connect('button_press_event', press)
    ax.figure.canvas.mpl_connect('button_release_event', release)
    ax.figure.canvas.mpl_connect('motion_notify_event', motion)

def main():

    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description="Manually label images for training")
    parser.add_argument("--path", help="directory where images are located")
    parser.add_argument(
        "--datapath", help="(deprecated) directory where images are located"
    )
    # parser.add_argument(
    #     "--pole_length", help="length of pole in cm"
    # )
    parser.add_argument(
        "--subset_to_label", help="label every N images"
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
                    #cameraids.append(splitline[0])
                    filename.append(splitline[0])
                    creationTimes.append(splitline[1])
                    topX.append(splitline[2])
                    topY.append(splitline[3])
                    bottomX.append(splitline[4])
                    bottomY.append(splitline[5])
                    PixelLengths.append(splitline[6].strip("\n"))
                    #snowdepths.append(splitline[7].strip("\n"))

    except FileNotFoundError:
        write_headers_line = True
    if write_headers_line:
        print("labels.csv is corrupted or does not exist, creating...")
        with open(f"{args.path}/labels.csv", "w") as labels2_csv:
            labels2_csv.write(
                '"filename","datetime","x1","y1","x2","y2","PixelLengths"'
            )

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

        ax = plt.gca()
        enable_scroll_zoom_and_pan(ax)

        plt.title("label top and then bottom of 10cm section \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom", fontweight="bold")
        points = plt.ginput(3, timeout=0, mouse_pop=2)
        top_10, bottom_10 = points[0], points[1]
        plt.close()

        figure = plt.figure(figsize=(20, 10), num=Path(file).name)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax = plt.gca()
        enable_scroll_zoom_and_pan(ax)

        plt.title("label top and then bottom of full pole \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom.", fontweight="bold")
        points = plt.ginput(3, timeout=0, mouse_pop=2)
        top, bottom = points[0], points[1]
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

            ax = plt.gca()
            enable_scroll_zoom_and_pan(ax)

            plt.title("label top and then bottom of full pole \n Click ANYWHERE to confirm | BACKSPACE to undo | RIGHT-CLICK drag | SCROLL zoom.", fontweight="bold")
            points = plt.ginput(3, timeout=0, mouse_pop=2)
            top, bottom = points[0], points[1]
            topX.append(top[0]), topY.append(top[1])
            bottomX.append(bottom[0]), bottomY.append(bottom[1])
            plt.close()

            PixelLength = math.dist(top, bottom)
            PixelLengths.append(PixelLength)

            ## save data to labels.csv
            nextline = f"\n{Path(file).name},{os.path.getctime(file)},{top[0]},{top[1]},{bottom[0]},{bottom[1]},{PixelLength}"
            with open(f"{args.path}/labels.csv", "a") as labels2_csv:
                labels2_csv.write(nextline)

            filename.append(Path(file).name)
            creationTime = os.path.getmtime(file)
            dt_c = datetime.datetime.fromtimestamp(creationTime)
            formatted_datetime = dt_c.strftime("%m/%d/%Y %H:%M")
            creationTimes.append(formatted_datetime)

            ## snowdepth ##
            snowdepth = pole_length_cm_lookup[cameraID] - (PixelLength * conversion_lookup[cameraID])
            snowdepths.append(snowdepth)

        i += 1

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
