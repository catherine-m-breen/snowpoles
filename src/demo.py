'''
written by Catherine Breen 
July 1, 2024

load model and run on example images
export the csv of pixel lengths and snow depths

example command line to run:

python src/demo.py

'''

import tomllib
with open("config.toml", "rb") as configfile:
    config = tomllib.load(configfile)

if config["extras"]["verbose"]: print("[1/12] Importing torch...")
import torch

if config["extras"]["verbose"]: print("[2/12] Importing numpy...")
import numpy as np

if config["extras"]["verbose"]: print("[3/12] Importing cv2...")
import cv2

if config["extras"]["verbose"]: print("[4/12] Importing snowPoleResNet50...")
from model import snowPoleResNet50

if config["extras"]["verbose"]: print("[5/12] Importing glob...")
import glob

if config["extras"]["verbose"]: print("[6/12] Importing pandas...")
import pandas as pd

if config["extras"]["verbose"]: print("[7/12] Importing tqdm...")
from tqdm import tqdm

if config["extras"]["verbose"]: print("[8/12] Importing matplotlib...")
import matplotlib.pyplot as plt

if config["extras"]["verbose"]: print("[9/12] Importing distance...")
from scipy.spatial import distance

if config["extras"]["verbose"]: print("[10/12] Importing os...")
import os

if config["extras"]["verbose"]: print("[11/12] Importing Path...")
from pathlib import Path

if config["extras"]["verbose"]: print("[12/12] Importing download_models...")
from model_download import download_models

# Comment out this line to disable dark mode
plt.style.use("./themes/dark.mplstyle")

def vis_predicted_keypoints(file, image, keypoints, color=(0, 255, 0), diameter=15):
    file = file.split(".")[0].replace("\\", "/")
    output_keypoint = keypoints.reshape(-1, 2)
    plt.imshow(image)
    for p in range(output_keypoint.shape[0]):
        if p == 0: 
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## top
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## bottom
    figpath = f"demo_predictions/pred_{file}.png"
    if not os.path.exists("/".join(figpath.split("/")[:-1])):
        os.makedirs("/".join(figpath.split("/")[:-1]))
    plt.savefig(f"demo_predictions/pred_{file}.png")
    plt.close()

def load_model(device):
    ## this will most likely be cpu for most users, unless using a GPU
    
    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(device)

    # pull model from cloud storage
    model_path = "models/CO_and_WA_model.pth"

    torch.serialization.add_safe_globals([torch.nn.modules.loss.SmoothL1Loss])
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    # load model weights state_dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict(model, device): ## 
 
    if not os.path.exists(f"demo_predictions"):
        os.makedirs(f"demo_predictions", exist_ok=True)

    Cameras, filenames = [], []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    total_length_pixels = []
    snow_depths = []
  
    snowpolefiles = glob.glob(f"example_data/**/*")
    ## full length of poles in cm
    metadata = pd.read_csv(f"example_data/pole_metadata.csv")
    
    with torch.no_grad():
        for i, file in tqdm(enumerate(snowpolefiles)):

            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, *_ = image.shape
            image = cv2.resize(image, (224,224))
            image = image / 255.0   

            # again reshape to add grayscale channel format
            filename = file.replace("\\", "/").split('/')[-1]
            Camera = filename.split('_')[0]
            
            ## add an empty dimension for sample size
            image = np.transpose(image, (2, 0, 1)) ## 
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0)
            image = image.to(device)

            #######
            outputs = model(image)
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

            vis_predicted_keypoints(filename, image, pred_keypoint,) 
            x1_pred, y1_pred, x2_pred, y2_pred = pred_keypoint[0], pred_keypoint[1], pred_keypoint[2], pred_keypoint[3]
            
            Cameras.append(Camera)
            filenames.append(filename)
            x1s_pred.append(x1_pred), y1s_pred.append(y1_pred), x2s_pred.append(x2_pred), y2s_pred.append(y2_pred)
            total_length_pixel = distance.euclidean([x1_pred,y1_pred],[x2_pred,y2_pred])
            total_length_pixels.append(total_length_pixel)

            ## snow depth conversion ## 
            for i, cam in enumerate(metadata["camera_id"]):
                if cam == Camera:
                    full_length_pole_cm = metadata['pole_length_cm'][i]
                    pixel_cm_conversion = metadata['pixel_cm_conversion'][i]
                    snow_depth = full_length_pole_cm - (pixel_cm_conversion * total_length_pixel)
                    snow_depths.append(snow_depth)

    results = pd.DataFrame({'camera_id':Cameras, 'filename':filenames, \
        'x1_pred': x1s_pred, 'y1s_pred': y1s_pred, 'x2_pred': x2s_pred, 'y2_pred': y2s_pred, \
                            'total_length_pixel': total_length_pixels, 'snow_depth':snow_depths})
    
    results.to_csv(f"demo_predictions/demo_results.csv")

    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    download_models()

    torch.serialization.add_safe_globals([torch.nn.modules.loss.SmoothL1Loss])
    model = load_model(device)

    ## returns a set of images of outputs
    outputs = predict(model, device)  

if __name__ == '__main__':
    main()



