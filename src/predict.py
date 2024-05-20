'''
load model and run on data points 
export the csv of the data points and just use the bottom

example command line to run:

python src/predict.py --image_path '/Volumes/CatBreen/CV4ecology/SNEX20_TLI_resized/CHE10' --output_path '/Volumes/CatBreen/Chewelah_resized'

python src/predict.py --image_path '/Volumes/CatBreen/Chewelah_resized/samples/CHE10' --output_path 'Volumes/CatBreen/Chewelah_resized'

python src/predict.py --image_path '/Users/catherinebreen/Documents/Chapter 1/WRRsubmission/data/448res/SNEX20_TLI_resized_clean/CHE2' --output_path '/Users/catherinebreen/Documents/Chapter 1/WRRsubmission/data/448res_results/CNNkeypoint'

python src/predict.py --model_path '/Users/catherinebreen/Documents/Chapter1/WRRsubmission/manuscript/model' --folder_path '/Users/catherinebreen/code/snow-Dayz/snowpoles/example_data/cam1' --output_path '/Users/catherinebreen/Documents' --dir_path '/Users/catherinebreen/code/snow-Dayz/snowpoles/example_data'


'''

import torch
import numpy as np
import cv2
import albumentations  ## may need to do pip install
import config
#import config_cpu as config
from model import snowPoleResNet50
import argparse
import glob
import IPython
import utils
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance

def load_model(args):
    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    ## katie's settings:
    if args.model_path == 'NULL':
        checkpoint = torch.load(config.OUTPUT_PATH + '/model_epoch50.pth', map_location=torch.device('mps')) ##config.OUTPUT_PATH
    else: #### Load the model based on how the user download the code folder to computer 
        checkpoint = torch.load(args.model_path + '/model_epoch50.pth', map_location=torch.device('cpu')) 
        # current_directory = os.getcwd()
        # neighboring_directory = os.path.join(current_directory, '/model_folder')
        # model_path = os.path.join(neighboring_directory, 'model_epoch50.pth')
        # checkpoint = torch.load(model_path, map_location=torch.device('mps'))
        
    #checkpoint = torch.load()
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict(model, args): ## try this without a dataloader
    #files =  glob.glob(args.image_path + ('/**/*.JPG'))
    #df_data = pd.read_csv(f"{config.ROOT_PATH}/snowPoles_labels.csv")
 
    if not os.path.exists(f"{args.output_path}/predictions"):
        os.makedirs(f"{args.output_path}/predictions", exist_ok=True)

    Cameras, filenames = [], []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    total_length_pixels = []
    
    snowpolefiles1 = glob.glob(f"{args.folder_path}/*")
    snowpolefiles2 = glob.glob(f"{args.dir_path}/**/*")
    
    if args.dir_path != '/example_data':
        snowpolefiles = snowpolefiles2
    else:
        snowpolefiles = snowpolefiles1

    #num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, file in tqdm(enumerate(snowpolefiles)): #, total=num_batches):
      
            ## because not using the dataset.py we will do it manually ##
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, *_ = image.shape
            #IPython.embed()
            image = cv2.resize(image, (224,224))
            image = image / 255.0   

           # again reshape to add grayscale channel format
            filename = file.split('/')[-1]
            Camera = filename.split('_')[0]
            
            ## add an empty dimension for sample size
            image = np.transpose(image, (2, 0, 1)) ## to get channels to line up
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0)
            image = image.to(config.DEVICE)

            #######
            outputs = model(image)
            ## below is adjusted for M1 max
            outputs = outputs.cpu().numpy() #outputs.detach().numpy()
            #output_list.append(outputs)
            pred_keypoint = np.array(outputs[0], dtype='float32')

            #### rescale to 448 x 448 because that is what the images are stored in: 
            ###########
            image = image.squeeze()
            image = image.cpu()
            image = np.transpose(image, (1, 2, 0))
            image = np.array(image, dtype='float32')
            #image = cv2.resize(image, (448,448))
            #pred_keypoint = pred_keypoint * [448 / 224] #keypoints * [self.resize / orig_w, self.resize / orig_h]
            IPython.embed()
            image = cv2.resize(image, (w, h))
            pred_keypoint[0] = pred_keypoint[0] * (w / 224)
            pred_keypoint[2] = pred_keypoint[2] * (w /224)
            pred_keypoint[1] = pred_keypoint[1] * (h / 224)
            pred_keypoint[3] = pred_keypoint[3] * (h /224)
            ###########

            utils.vis_predicted_keypoints(args, filename, image, pred_keypoint,) ## visualize points
            x1_pred, y1_pred, x2_pred, y2_pred = pred_keypoint[0], pred_keypoint[1], pred_keypoint[2], pred_keypoint[3]
            
            Cameras.append(Camera)
            filenames.append(filename)
            x1s_pred.append(x1_pred), y1s_pred.append(y1_pred), x2s_pred.append(x2_pred), y2s_pred.append(y2_pred)
                ## error
            total_length_pixel = distance.euclidean([x1_pred,y1_pred],[x2_pred,y2_pred])
            total_length_pixels.append(total_length_pixel)

    #IPython.embed()
    results = pd.DataFrame({'Camera':Cameras, 'filename':filenames, \
        'x1_pred': x1s_pred, 'y1s_pred': y1s_pred, 'x2_pred': x2s_pred, 'y2_pred': y2s_pred, 'total_length_pixel': total_length_pixels})

    results.to_csv(f"{args.output_path}/predictions/{Camera}_results.csv")

    return results

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Predict top and bottom coordinates.')
    parser.add_argument('--model_path', required=False, help = 'Path to model', default = 'NULL')
    parser.add_argument('--dir_path', required=False, help='Path to camera image directory', default = '/example_data')
    parser.add_argument('--folder_path', required=False, help='Path to camera image folder', default = "/example_data/cam1")
    parser.add_argument('--output_path', required=True, help='Path to output folder', default = "/example_data")
    args = parser.parse_args()


    #args = parser.parse_args()
    model = load_model(args)

    ## returns a set of images of outputs
    outputs = predict(model, args)  

    #results = eval(outputs)

if __name__ == '__main__':
    main()



