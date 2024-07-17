'''
load model and run on data points 
export the csv of the data points and just use the bottom

example command line to run:

python src/demo.py

'''

import torch
import numpy as np
import cv2
from model import snowPoleResNet50
import argparse
import glob
import utils
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
import os

def download_models(): 
  ## check if path exists 
  if not os.path.exists(f"~/models"):
        os.makedirs(f"~/models", exist_ok=True)
        url = 'insert zenodo link'
        ## downloaded 
  return print('models download')

def vis_predicted_keypoints(file, image, keypoints, color=(0,255,0), diameter=15):
    output_keypoint = keypoints.reshape(-1, 2)
    plt.imshow(image)
    for p in range(output_keypoint.shape[0]):
        if p == 0: 
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## top
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## bottom
    plt.savefig(f"~/predictions/image_{file}.png")
    plt.close()

def load_model(args):
    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)

    ## this will most likely be cpu for most users, unless using a GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pull model from cloud storage
    model_path = '~/CO_and_WA_model.pth' 
    
    checkpoint = torch.load(model_path, map_location=torch.device(device)) 
    
  # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict(model, args): ## 
 
    if not os.path.exists(f"~/predictions"):
        os.makedirs(f"~/predictions", exist_ok=True)

    Cameras, filenames = [], []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    total_length_pixels = []
  
    snowpolefiles = glob.glob(f"~/example_data/**/*")
    
    with torch.no_grad():
        for i, file in tqdm(enumerate(snowpolefiles)): 
    
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, *_ = image.shape
            image = cv2.resize(image, (224,224))
            image = image / 255.0   

            # again reshape to add grayscale channel format
            filename = file.split('/')[-1]
            Camera = filename.split('_')[0]
            
            ## add an empty dimension for sample size
            image = np.transpose(image, (2, 0, 1)) ## 
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0)
            image = image.to(config.DEVICE)

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
            ###########

            vis_predicted_keypoints(filename, image, pred_keypoint,) 
            x1_pred, y1_pred, x2_pred, y2_pred = pred_keypoint[0], pred_keypoint[1], pred_keypoint[2], pred_keypoint[3]
            
            Cameras.append(Camera)
            filenames.append(filename)
            x1s_pred.append(x1_pred), y1s_pred.append(y1_pred), x2s_pred.append(x2_pred), y2s_pred.append(y2_pred)
            total_length_pixel = distance.euclidean([x1_pred,y1_pred],[x2_pred,y2_pred])
            total_length_pixels.append(total_length_pixel)

            ## snow depth conversion ## 
            

    results = pd.DataFrame({'Camera':Cameras, 'filename':filenames, \
        'x1_pred': x1s_pred, 'y1s_pred': y1s_pred, 'x2_pred': x2s_pred, 'y2_pred': y2s_pred, 'total_length_pixel': total_length_pixels})

    results.to_csv(f"{args.output_path}/predictions/{Camera}_results.csv")

    return results

def main():
    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description='Predict top and bottom coordinates.')
    parser.add_argument('--dir_path', required=False, help='Path to camera image directory', default = '/example_data')
  
    args = parser.parse_args()

    download_models()
    #args = parser.parse_args()
    model = load_model(args)

    ## returns a set of images of outputs
    outputs = predict(model, args)  

if __name__ == '__main__':
    main()



