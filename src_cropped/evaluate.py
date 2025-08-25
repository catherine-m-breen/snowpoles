'''
load model and run on data points with crop support
export the csv of the data points and just use the bottom

example command line to run:
(make sure config file is set to the right model!)
python src/evaluate.py
'''

# Import startup libraries
import argparse
import tomli as tomllib
import os

# [Keep your existing argument parser and config loading code here...]

import torch
import numpy as np
import config
from model import snowPoleResNet50
import IPython
import utils
import pandas as pd
from dataset import train_data, valid_data
from tqdm import tqdm
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt

def load_model():
    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(args.device)
    # load the model checkpoint
    model_path = f"{args.model}/model.pth"
    checkpoint = torch.load(model_path, map_location=torch.device(args.device))
    print(f"loading model from the following path: {args.model}")
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def transform_keypoints_to_original(keypoints_crop, crop_params, original_size=(448, 448), crop_resize=224):
    """
    Transform keypoints from crop coordinates back to original image coordinates
    
    Args:
        keypoints_crop: Keypoints in crop coordinate system (after resize to 224x224)
        crop_params: (y_start, y_end, x_start, x_end) of the crop
        original_size: Original image size (height, width)
        crop_resize: Size the crop was resized to (224)
    """
    if crop_params is None:
        # No crop was applied, just scale from resize back to original
        scale_x = original_size[1] / crop_resize  # width scaling
        scale_y = original_size[0] / crop_resize  # height scaling
        return keypoints_crop * [scale_x, scale_y]
    
    y_start, y_end, x_start, x_end = crop_params
    crop_height = y_end - y_start
    crop_width = x_end - x_start
    
    # Scale from resized crop (224x224) back to original crop size
    scale_x = crop_width / crop_resize
    scale_y = crop_height / crop_resize
    
    keypoints_original_crop = keypoints_crop * [scale_x, scale_y]
    
    # Transform from crop coordinates back to original image coordinates
    keypoints_original = keypoints_original_crop + [x_start, y_start]
    
    return keypoints_original

def predict(model, data, eval='eval'): 

    if not os.path.exists(f"{args.output}/{eval}"):
        os.makedirs(f"{args.output}/{eval}", exist_ok=True)

    # Get crop parameters from config or dataset
    crop_params = None
    if hasattr(data.dataset, 'crop_params') and data.dataset.crop_params is not None:
        crop_params = data.dataset.crop_params
        print(f"Using crop parameters: {crop_params}")
    
    output_list = []
    Cameras, filenames = [], []
    x1s_true, y1s_true, x2s_true, y2s_true = [], [], [], []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    # Store both crop and original coordinates
    x1s_true_orig, y1s_true_orig, x2s_true_orig, y2s_true_orig = [], [], [], []
    x1s_pred_orig, y1s_pred_orig, x2s_pred_orig, y2s_pred_orig = [], [], [], []
    
    top_pixel_errors, bottom_pixel_errors, total_length_pixels = [], [], []
    total_length_pixel_actuals = []
    mape_errors = []
    mape_errors_sd = []
    mape_errors_sd_clean = []

    automated_sds, manual_sds, diff_sds = [], [], []

    metadata = pd.read_csv(f"{config.metadata}")
    labels = pd.read_csv(f"{config.labels}")

    with torch.no_grad():
        for i, data_batch in tqdm(enumerate(data)): 
            image, keypoints = data_batch['image'].to(args.device), data_batch['keypoints'].to(config.DEVICE)
            filename = data_batch['filename']
            Camera = filename.split('_W')[0]

            # Current keypoints are in crop coordinate system (224x224)
            keypoints_crop = keypoints.detach().cpu().numpy().reshape(-1,2)
            x1_true_crop, y1_true_crop, x2_true_crop, y2_true_crop = keypoints_crop[0,0], keypoints_crop[0,1], keypoints_crop[1,0], keypoints_crop[1,1]
            
            # Transform true keypoints back to original image coordinates
            keypoints_true_orig = transform_keypoints_to_original(keypoints_crop, crop_params)
            x1_true_orig, y1_true_orig, x2_true_orig, y2_true_orig = keypoints_true_orig[0,0], keypoints_true_orig[0,1], keypoints_true_orig[1,0], keypoints_true_orig[1,1]
            
            ## add an empty dimension for sample size
            image = image.unsqueeze(0)
            outputs = model(image)
            outputs = outputs.detach().cpu().numpy()
            
            # Predicted keypoints are in crop coordinate system (224x224)
            pred_keypoint_crop = np.array(outputs[0], dtype='float32').reshape(-1, 2)
            x1_pred_crop, y1_pred_crop, x2_pred_crop, y2_pred_crop = pred_keypoint_crop[0,0], pred_keypoint_crop[0,1], pred_keypoint_crop[1,0], pred_keypoint_crop[1,1]
            
            # Transform predicted keypoints back to original image coordinates
            pred_keypoint_orig = transform_keypoints_to_original(pred_keypoint_crop, crop_params)
            x1_pred_orig, y1_pred_orig, x2_pred_orig, y2_pred_orig = pred_keypoint_orig[0,0], pred_keypoint_orig[0,1], pred_keypoint_orig[1,0], pred_keypoint_orig[1,1]
            
            # Visualize with crop coordinates (for consistency with training)
            utils.eval_keypoints_plot(filename, image, outputs, eval, orig_keypoints=keypoints_crop)
            
            Cameras.append(Camera)
            filenames.append(filename)
            
            # Store crop coordinates (for internal consistency)
            x1s_true.append(x1_true_crop), y1s_true.append(y1_true_crop), x2s_true.append(x2_true_crop), y2s_true.append(y2_true_crop)
            x1s_pred.append(x1_pred_crop), y1s_pred.append(y1_pred_crop), x2s_pred.append(x2_pred_crop), y2s_pred.append(y2_pred_crop)
            
            # Store original coordinates (for real-world measurements)
            x1s_true_orig.append(x1_true_orig), y1s_true_orig.append(y1_true_orig), x2s_true_orig.append(x2_true_orig), y2s_true_orig.append(y2_true_orig)
            x1s_pred_orig.append(x1_pred_orig), y1s_pred_orig.append(y1_pred_orig), x2s_pred_orig.append(x2_pred_orig), y2s_pred_orig.append(y2_pred_orig)
            
            ## Use ORIGINAL coordinates for real-world measurements
            total_length_pixel = distance.euclidean([x1_pred_orig, y1_pred_orig], [x2_pred_orig, y2_pred_orig])
            
            try: 
                full_length_pole_cm = metadata[metadata['camera_id'] == Camera]['pole_length_cm'].values[0]
                pixel_cm_conversion = metadata[metadata['camera_id'] == Camera]['pixel_cm_conversion'].values[0] 
                automated_sd = full_length_pole_cm - (pixel_cm_conversion * total_length_pixel)
            
            except Exception: 
                print(Camera)
                IPython.embed()
            automated_sds.append(automated_sd)

            # Manual measurements should also be in original coordinates
            manual_pixel_length = labels[labels['filename'] == filename]['PixelLengths'].values[0]
            manual_snowdepth = full_length_pole_cm - (pixel_cm_conversion * manual_pixel_length)
            difference = manual_snowdepth - automated_sd
            manual_sds.append(manual_snowdepth), diff_sds.append(difference)

            ## Calculate errors using ORIGINAL coordinates for real-world accuracy
            top_pixel_error = distance.euclidean([x1_true_orig, y1_true_orig], [x1_pred_orig, y1_pred_orig])
            bottom_pixel_error = distance.euclidean([x2_true_orig, y2_true_orig], [x2_pred_orig, y2_pred_orig])
            total_length_pixel_actual = distance.euclidean([x1_true_orig, y1_true_orig], [x2_true_orig, y2_true_orig])

            # MAPE
            mape_error = utils.MAPE(total_length_pixel_actual, total_length_pixel)
            mape_error_sd = utils.MAPE(manual_snowdepth, automated_sd)
            mape_errors_sd.append(mape_error_sd)
            
            top_pixel_errors.append(top_pixel_error), bottom_pixel_errors.append(bottom_pixel_error), total_length_pixels.append(total_length_pixel)
            total_length_pixel_actuals.append(total_length_pixel_actual), mape_errors.append(mape_error)
    
    # Save results with both crop and original coordinates
    results = pd.DataFrame({
        'Camera': Cameras, 'filename': filenames, 
        # Crop coordinates
        'x1_true_crop': x1s_true, 'y1_true_crop': y1s_true, 'x2_true_crop': x2s_true, 'y2_true_crop': y2s_true,
        'x1_pred_crop': x1s_pred, 'y1_pred_crop': y1s_pred, 'x2_pred_crop': x2s_pred, 'y2_pred_crop': y2s_pred,
        # Original coordinates  
        'x1_true_orig': x1s_true_orig, 'y1_true_orig': y1s_true_orig, 'x2_true_orig': x2s_true_orig, 'y2_true_orig': y2s_true_orig,
        'x1_pred_orig': x1s_pred_orig, 'y1_pred_orig': y1s_pred_orig, 'x2_pred_orig': x2s_pred_orig, 'y2_pred_orig': y2s_pred_orig,
        # Errors and measurements
        'top_pixel_error': top_pixel_errors, 'bottom_pixel_error': bottom_pixel_errors, 
        'total_length_pixel': total_length_pixels, 'total_length_pixel_actual': total_length_pixel_actuals,
        'automated_depth': automated_sds, 'manual_snowdepth': manual_sds, 'difference': diff_sds, 
        'mape': mape_errors, 'mape_sd': mape_errors_sd
    })

    results.to_csv(f"{args.output}/{eval}/indiv_img_eval_results.csv")
    
    # [Keep your existing statistics printing and saving code...]
    
    return results

def main():
    model = load_model()
    print('results on valid data\n')
    outputs = predict(model, valid_data)

if __name__ == '__main__':
    main()