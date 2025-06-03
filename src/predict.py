"""
written by: Catherine Breen
July 1, 2024

Training script for users to fine tune model from Breen et. al 2024
Please cite: 

Breen, C. M., Currier, W. R., Vuyovich, C., Miao, Z., & Prugh, L. R. (2024). 
Snow Depth Extraction From Time‐Lapse Imagery Using a Keypoint Deep Learning Model. 
Water Resources Research, 60(7), e2023WR036682. https://doi.org/10.1029/2023WR036682

Example run:
python src/predict.py --model_path './output1/model.pth' --img_dir './nontrained_data'  --metadata './nontrained_data/pole_metadata.csv'


"""

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
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance


def download_models():
    """
    see the Zenodo page for the latest models
    """
    root = os.getcwd()
    save_path = f"{root}/models"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    url = "https://zenodo.org/records/12764696/files/CO_and_WA_model.pth"

    # download if does not exist
    if not os.path.exists(f"{save_path}/CO_and_WA_model.pth"):
        wget_command = f"wget {url} -P {save_path}"
        os.system(wget_command)
        return print("\n models download! \n")
    else:
        return print("model already saved")


def vis_predicted_keypoints(file, image, keypoints, color=(0, 255, 0), diameter=15):
    file = file.split(".")[0]
    output_keypoint = keypoints.reshape(-1, 2)
    plt.imshow(image)
    for p in range(output_keypoint.shape[0]):
        if p == 0:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], "r.")  ## top
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], "r.")  ## bottom
    plt.savefig(f"predictions/pred_{file}.png")
    plt.close()


def load_model(args):
    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    if args.model_path == "models/CO_and_WA_model.pth":  ## uses model from paper
        model_path = "models/CO_and_WA_model.pth"
        checkpoint = torch.load(model_path, map_location=torch.device(config.DEVICE))
    else:  ## your customized model
        checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(model, args, device):  ##

    if not os.path.exists(f"predictions"):
        os.makedirs(f"predictions", exist_ok=True)

    Cameras, filenames = [], []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    total_length_pixels = []
    snow_depths = []

    ## folder or directory
    snowpolefiles1 = glob.glob(f"{args.img_folder}/*")
    snowpolefiles2 = glob.glob(f"{args.img_dir}/**/*")

    ## checks for a directory
    if args.img_dir != "/example_data":
        snowpolefiles = snowpolefiles2
    else:  # assumes it is a camerafolder
        snowpolefiles = snowpolefiles1

    metadata = pd.read_csv(f"{args.metadata}")

    with torch.no_grad():
        for i, file in tqdm(enumerate(snowpolefiles)):

            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, *_ = image.shape
            image = cv2.resize(image, (224, 224))
            image = image / 255.0

            # again reshape to add grayscale channel format
            filename = file.split("/")[-1]
            Camera = filename.split("/")[
                -1
            ]  ## assumes in a folder with camera name ('cam1', 'cam2', etc)

            ## add an empty dimension for sample size
            image = np.transpose(image, (2, 0, 1))  ##
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0)
            image = image.to(device)

            #######
            outputs = model(image)
            outputs = outputs.cpu().numpy()
            pred_keypoint = np.array(outputs[0], dtype="float32")

            image = image.squeeze()
            image = image.cpu()
            image = np.transpose(image, (1, 2, 0))
            image = np.array(image, dtype="float32")

            ## resize back up to original size and project predicted points onto original size
            image = cv2.resize(image, (w, h))
            pred_keypoint[0] = pred_keypoint[0] * (w / 224)
            pred_keypoint[2] = pred_keypoint[2] * (w / 224)
            pred_keypoint[1] = pred_keypoint[1] * (h / 224)
            pred_keypoint[3] = pred_keypoint[3] * (h / 224)

            vis_predicted_keypoints(
                filename,
                image,
                pred_keypoint,
            )
            x1_pred, y1_pred, x2_pred, y2_pred = (
                pred_keypoint[0],
                pred_keypoint[1],
                pred_keypoint[2],
                pred_keypoint[3],
            )

            Cameras.append(Camera)
            filenames.append(filename)
            x1s_pred.append(x1_pred), y1s_pred.append(y1_pred), x2s_pred.append(
                x2_pred
            ), y2s_pred.append(y2_pred)
            total_length_pixel = distance.euclidean(
                [x1_pred, y1_pred], [x2_pred, y2_pred]
            )
            total_length_pixels.append(total_length_pixel)

            ## snow depth conversion ##
            try:
                full_length_pole_cm = metadata[metadata["camera_id"] == Camera][
                    "pole_length_cm"
                ].values[0]
                pixel_cm_conversion = metadata[metadata["camera_id"] == Camera][
                    "pixel_cm_conversion"
                ].values[0]
                snow_depth = full_length_pole_cm - (
                    pixel_cm_conversion * total_length_pixel
                )
                snow_depths.append(snow_depth)
            except:
                ## if you don't have a metadata stored properly it will just insert a 0 for snowdepth
                snow_depths.append(0)

    results = pd.DataFrame(
        {
            "camera_id": Cameras,
            "filename": filenames,
            "x1_pred": x1s_pred,
            "y1_pred": y1s_pred,
            "x2_pred": x2s_pred,
            "y2_pred": y2s_pred,
            "total_length_pixel": total_length_pixels,
            "snow_depth": snow_depths,
        }
    )

    results.to_csv(f"predictions/results.csv")

    return results


def main():
    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description="Predict top and bottom coordinates.")
    parser.add_argument(
        "--model_path",
        required=False,
        help="Path to model",
        default="models/CO_and_WA_model.pth",
    )
    parser.add_argument(
        "--img_dir",
        required=False,
        help="Path to camera image directory",
        default="/example_data",
    )
    parser.add_argument(
        "--img_folder",
        required=False,
        help="Path to camera image folder",
        default="/example_data/cam1",
    )
    parser.add_argument(
        "--metadata",
        required=False,
        help="Path to pole metadata",
        default="/example_data/pole_metadata.csv",
    )
    args = parser.parse_args()

    model = load_model(args)
    device = "cpu"
    predict(model, args, device)


if __name__ == "__main__":
    main()
