'''
written by: Catherine Breen
July 1, 2024

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682

Example run:
python src/predict.py --model_path './output1/model.pth' --img_dir './nontrained_data'  --metadata './nontrained_data/pole_metadata.csv'


'''

# Import startup libraries
import argparse
import os
import tomli as tomllib
from pathlib import Path

# for datetime
import datetime

# for predict
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
import torch
from tqdm import tqdm
import IPython
import json

def vis_predicted_keypoints(file, image, keypoints, color=(0, 255, 0), diameter=15):
    import matplotlib.pyplot as plt
    #file = file.split(".")[0]
    file = Path(file).stem  
    output_keypoint = keypoints.reshape(-1, 2)
    plt.imshow(image)
    for p in range(output_keypoint.shape[0]):
        if p == 0: 
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## top
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## bottom
    plt.savefig(f"{args.model}/predictions/pred_{file}.png")
    plt.close()

def load_camera_mapping(model_path):
    """
    Load camera mapping from the saved JSON file
    """
    mapping_file = os.path.join(model_path, "camera_mapping.json")
    with open(mapping_file, 'r') as f:
        camera_mapping = json.load(f)
    print(f"Loaded camera mapping with {len(camera_mapping)} cameras from {mapping_file}")
    return camera_mapping

def load_model(args):
    from model import snowPoleResNet50
    import torch

    camera_mapping = load_camera_mapping(args.model)
    num_cameras = len(camera_mapping)
    
    # Initialize model with time embedding support
    model = snowPoleResNet50(
        pretrained=False, 
        requires_grad=False, 
        num_cameras=num_cameras,
        embedding_dim=64,  # Should match training
        time_embedding_dim=32  # Add time embedding dimension
    ).to(args.device)
    
    # load the model checkpoint
    torch.serialization.add_safe_globals([torch.nn.modules.loss.SmoothL1Loss])
    model_path = f"{args.model}/model.pth"
    checkpoint = torch.load(model_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, camera_mapping

def format_datetime(dt_string):
    """
    Format datetime string to mm/dd/yy HH:MM format used in training
    """
    dt = datetime.datetime.strptime(dt_string, "%m/%d/%Y %H:%M")
    return dt.strftime("%m/%d/%y %H:%M")

def predict(model, args, device, camera_mapping = None):  
    if not os.path.exists(f"{args.model}/predictions"):
        os.makedirs(f"{args.model}/predictions", exist_ok=True)

    Cameras, filenames = [], []
    datetimes = []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    total_length_pixels = []
    snow_depths = []

    ## folder or directory
    snowpolefiles = list(Path(args.path).rglob("*.JPG"))

    metadata = pd.read_csv(f"{args.path}/pole_metadata.csv")

    with torch.no_grad():
        for i, file in tqdm(enumerate(snowpolefiles)): 
    
            image = cv2.imread(str(file))
            creationTime = os.path.getmtime(file)
            dt_c = datetime.datetime.fromtimestamp(creationTime)
            formatted_datetime = dt_c.strftime("%m/%d/%y %H:%M")  # Changed to %y for 2-digit year
            datetimes.append(formatted_datetime)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, *_ = image.shape
            image = cv2.resize(image, (224,224))
            image = image / 255.0   

            # again reshape to add grayscale channel format
            file_path = Path(file)
            filename = file_path.name
            Camera = file_path.stem.split('_')[0]
            
            ## add an empty dimension for sample size
            image = np.transpose(image, (2, 0, 1)) ## 
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0)
            image = image.to(device)

            ## camera mapping
            camera_id = camera_mapping[Camera]
            camera_ids = torch.tensor([camera_id], dtype=torch.long, device=device)
            
            ## datetime for time embedding
            datetime_strings = [formatted_datetime]  # List with single datetime string

            ## Forward pass with datetime
            outputs = model(image, camera_ids, datetime_strings)
            outputs = outputs.cpu().numpy() 
            pred_keypoint = np.array(outputs[0], dtype='float32')

            image = image.squeeze()
            image = image.cpu()
            image = np.transpose(image, (1, 2, 0))
            image = np.array(image, dtype='float32')

            ## resize back up to original size and project predicted points onto original size
            image = cv2.resize(image, (w, h))
            pred_keypoint[0] = pred_keypoint[0] * (w / 224)
            pred_keypoint[2] = pred_keypoint[2] * (w /224)
            pred_keypoint[1] = pred_keypoint[1] * (h / 224)
            pred_keypoint[3] = pred_keypoint[3] * (h /224)

            if i % 100 == 0: vis_predicted_keypoints(filename, image, pred_keypoint,) 
            x1_pred, y1_pred, x2_pred, y2_pred = pred_keypoint[0], pred_keypoint[1], pred_keypoint[2], pred_keypoint[3]
            
            Cameras.append(Camera)
            filenames.append(filename)
            x1s_pred.append(x1_pred), y1s_pred.append(y1_pred), x2s_pred.append(x2_pred), y2s_pred.append(y2_pred)
            total_length_pixel = distance.euclidean([x1_pred,y1_pred],[x2_pred,y2_pred])
            total_length_pixels.append(total_length_pixel)

            ## snow depth conversion ## 
            try: 
                full_length_pole_cm = metadata[metadata['camera_id'] == Camera]['pole_length_cm'].values[0]
                pixel_cm_conversion = metadata[metadata['camera_id'] == Camera]['pixel_cm_conversion'].values[0] 
                snow_depth = full_length_pole_cm - (pixel_cm_conversion * total_length_pixel)
                snow_depths.append(snow_depth)
            except Exception as e: 
                print(f"Warning: Could not calculate snow depth for {Camera}: {e}")
                ## if you don't have a metadata stored properly it will just insert a 0 for snowdepth
                snow_depths.append(0)
            
    results = pd.DataFrame({'camera_id':Cameras, 'filename':filenames, 'datetime':datetimes, \
        'x1_pred': x1s_pred, 'y1_pred': y1s_pred, 'x2_pred': x2s_pred, 'y2_pred': y2s_pred, \
                            'total_length_pixel': total_length_pixels, 'snow_depth':snow_depths})
    
    results.to_csv(f"{args.model}/predictions/results.csv")

    return results

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Use a model to predict snow depth")
    parser.add_argument(
        "--model",
        required=False,
        help="model to use",
    )
    parser.add_argument("--path", help="directory where images are located")
    parser.add_argument(
        "--device", required=False, help='device to use for processing ("cpu" or "cuda")'
    )
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
    if not args.model:
        args.model = config["paths"]["models_output"]
    if not args.path:
        args.path = config["paths"]["input_images"]
    if not args.device:
        args.device = config["training"]["device"]
    if not args.output:
        args.output = config["paths"]["models_output"]

    # Confirmation
    if not args.no_confirm:
        print(
            "\n\n# The following options were specified in config.toml or as arguments:\n"
        )
        if (args.model.startswith("/")):
            print(
                "Model to use:\n"
                + str(args.model)
                + "\n"
            )
        else:
            print(
                "Model to use:\n"
                + os.getcwd()
                + "/"
                + str(args.model)
                + "\n"
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
        print("Device to use:\n" + args.device + "\n")
        if (args.output.startswith("/")):
            print(
                "Directory where marked images will be stored:\n"
                + str(args.output)
                + "\n"
            )
        else:
            print(
                "Directory where marked images will be stored:\n"
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


    # Import all libraries
    import albumentations
    import IPython
    import utils

    model, camera_mapping = load_model(args)
    device = 'cpu'
    predict(model, args, device, camera_mapping)  

if __name__ == '__main__':
    main()