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

# for predict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
import torch
from tqdm import tqdm
import IPython

from model import snowPoleResNet50
import torch


def vis_predicted_keypoints(config, file, image, keypoints, color=(0, 255, 0), diameter=15):
    file = Path(file).stem  
    output_keypoint = keypoints.reshape(-1, 2)
    plt.imshow(image)
    for p in range(output_keypoint.shape[0]):
        if p == 0: 
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## top
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## bottom
    plt.savefig(f"{config['paths']['models_output']}/predictions/pred_{file}.png")
    plt.close()

def load_model(config):
    model = snowPoleResNet50(pretrained=False, 
                             hidden_size=256, 
                             num_layers=2,     
                             num_classes=4,   ## could adjust and predict more poles in the image i guess? 
                             requires_grad=False).to(config['training']['device'])
    # load the model checkpoint
    #torch.serialization.add_safe_globals([torch.nn.modules.loss.SmoothL1Loss])
    model_path = f"{config['paths']['models_output']}/model.pth"
    checkpoint = torch.load(model_path, map_location=torch.device(config['training']['device']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict(model, config):  
    device = config['training']['device']

    if not os.path.exists(f"{config['paths']['models_output']}/predictions"):
        os.makedirs(f"{config['paths']['models_output']}/predictions", exist_ok=True)

    Cameras, filenames = [], []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    total_length_pixels = []
    snow_depths = []

    ## folder or directory
    #IPython.embed()
    #snowpolefiles = glob.glob(f"{args.path}/**/*")
    snowpolefiles = list(Path(config['paths']['input_images']).rglob("*.JPG"))
    metadata = pd.read_csv(f"{config['paths']['input_images']}/pole_metadata.csv")

    ## create sequences for predictions ## 
    sequence_length = 3 

    # Group files by parent directory using dictionary comprehension
    parent_dirs = set(file.parent for file in snowpolefiles)
    grouped_files = {parent: [f for f in snowpolefiles if f.parent == parent] for parent in parent_dirs}

    # Create sequences from each group
    X_sequences = []
    for parent_dir, files in grouped_files.items():
        # Sort files by name (assuming filenames have timestamps or sequential naming)
        sorted_files = sorted(files, key=lambda x: x.name)
        
        # Create overlapping sequences
        for i in range(len(sorted_files) - sequence_length + 1):
            seq = sorted_files[i:i + sequence_length]
            X_sequences.append(seq) 

    with torch.no_grad():
        for i, sequence in tqdm(enumerate(X_sequences)): 

            sequence_images = []
            for file_path in sequence:
                image = cv2.imread(str(file_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, *_ = image.shape
                image = cv2.resize(image, (224,224))
                image = image / 255.0   
                image = np.transpose(image, (2, 0, 1)) ##
                image = torch.tensor(image, dtype=torch.float)
                sequence_images.append(image)

            sequence_tensor = torch.stack(sequence_images)
            sequence_tensor = sequence_tensor.unsqueeze(0)
            sequence_tensor = sequence_tensor.to(device)

            outputs = model(sequence_tensor)
            outputs = outputs.cpu().numpy() 
            pred_keypoint = np.array(outputs[0], dtype='float32')

            ######## 
            last_file = sequence[-1]
            original_image = cv2.imread(str(last_file))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            h, w, *_ = original_image.shape
            
            file_path = Path(last_file)
            filename = file_path.name
            Camera = file_path.stem.split('_')[0]
            #### do i need this? 
            # image = np.transpose(image, (1, 2, 0))
            # image = np.array(image, dtype='float32')
            # image = cv2.resize(image, (w, h))
            ########

            ## resize back up to original size and project predicted points onto original size
            pred_keypoint[0] = pred_keypoint[0] * (w / 224)
            pred_keypoint[2] = pred_keypoint[2] * (w /224)
            pred_keypoint[1] = pred_keypoint[1] * (h / 224)
            pred_keypoint[3] = pred_keypoint[3] * (h /224)

            if i % 20 == 0: ## save every 20
                image = np.transpose(image, (1, 2, 0))
                image = np.array(image, dtype='float32')
                image = cv2.resize(image, (w, h))
                vis_predicted_keypoints(config, filename, image, pred_keypoint,) 
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
            except: 
                ## if you don't have a metadata stored properly it will just insert a 0 for snowdepth
                snow_depths.append(0)
            
    results = pd.DataFrame({'camera_id':Cameras, 'filename':filenames, \
        'x1_pred': x1s_pred, 'y1_pred': y1s_pred, 'x2_pred': x2s_pred, 'y2_pred': y2s_pred, \
                            'total_length_pixel': total_length_pixels, 'snow_depth':snow_depths})
    
    results.to_csv(f"{config['paths']['models_output']}/predictions/results.csv")

    return results

def main():
   
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='config_lstm.toml')
    args = parser.parse_args()

    print(f'Using config "{args.config}"')
    with open(args.config, "rb") as configfile:
        config = tomllib.load(configfile)

    model = load_model(config)
    predict(model, config)  

if __name__ == '__main__':
    main()



