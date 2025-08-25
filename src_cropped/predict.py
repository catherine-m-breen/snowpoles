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

def vis_predicted_keypoints(file, image, keypoints, color=(0, 255, 0), diameter=15):
    import matplotlib.pyplot as plt
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

def load_model(args):
    from model import snowPoleResNet50
    import torch

    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(args.device)
    # load the model checkpoint
    model_path = f"{args.model}/model.pth"
    checkpoint = torch.load(model_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_crop_params_from_config():
    """
    Get crop parameters from config file
    Returns (y_start, y_end, x_start, x_end) or None if no crop
    """
    try:
        with open("config.toml", "rb") as configfile:
            config = tomllib.load(configfile)
        
        if config['training'].get('use_crop', False):
            return (
                config['training']['crop_y_start'],
                config['training']['crop_y_end'], 
                config['training']['crop_x_start'],
                config['training']['crop_x_end']
            )
    except:
        pass
    
    return None

def transform_keypoints_to_original(keypoints_crop, crop_params, original_size=(448, 448), crop_resize=224):
    """
    Transform keypoints from crop coordinates back to original image coordinates
    
    Args:
        keypoints_crop: Keypoints in crop coordinate system (after resize to 224x224) - shape (4,)
        crop_params: (y_start, y_end, x_start, x_end) of the crop
        original_size: Original image size (height, width)
        crop_resize: Size the crop was resized to (224)
    
    Returns:
        keypoints_original: Keypoints in original image coordinates - shape (4,)
    """
    if crop_params is None:
        # No crop was applied, just scale from resize back to original
        scale_x = original_size[1] / crop_resize  # width scaling
        scale_y = original_size[0] / crop_resize  # height scaling
        keypoints_original = keypoints_crop.copy()
        keypoints_original[0] *= scale_x  # x1
        keypoints_original[2] *= scale_x  # x2
        keypoints_original[1] *= scale_y  # y1
        keypoints_original[3] *= scale_y  # y2
        return keypoints_original
    
    y_start, y_end, x_start, x_end = crop_params
    crop_height = y_end - y_start
    crop_width = x_end - x_start
    
    # Scale from resized crop (224x224) back to original crop size
    scale_x = crop_width / crop_resize
    scale_y = crop_height / crop_resize
    
    keypoints_original_crop = keypoints_crop.copy()
    keypoints_original_crop[0] *= scale_x  # x1
    keypoints_original_crop[2] *= scale_x  # x2
    keypoints_original_crop[1] *= scale_y  # y1
    keypoints_original_crop[3] *= scale_y  # y2
    
    # Transform from crop coordinates back to original image coordinates
    keypoints_original = keypoints_original_crop.copy()
    keypoints_original[0] += x_start  # x1
    keypoints_original[2] += x_start  # x2
    keypoints_original[1] += y_start  # y1
    keypoints_original[3] += y_start  # y2
    
    return keypoints_original

def predict(model, args, device):  
    if not os.path.exists(f"{args.model}/predictions"):
        os.makedirs(f"{args.model}/predictions", exist_ok=True)

    # Get crop parameters
    crop_params = (50, 450, 150, 250)  # (y_start, y_end, x_start, x_end)
    #get_crop_params_from_config()
    if crop_params:
        print(f"Using crop parameters: {crop_params}")
    else:
        print("No crop parameters found - processing full images")

    Cameras, filenames = [], []
    datetimes = []
    # Store both crop and original coordinates
    x1s_pred_crop, y1s_pred_crop, x2s_pred_crop, y2s_pred_crop = [], [], [], []
    x1s_pred_orig, y1s_pred_orig, x2s_pred_orig, y2s_pred_orig = [], [], [], []
    total_length_pixels = []
    snow_depths = []

    snowpolefiles = list(Path(args.path).rglob("*.JPG"))
    metadata = pd.read_csv(f"{args.path}/pole_metadata.csv")

    with torch.no_grad():
        for i, file in tqdm(enumerate(snowpolefiles)): 
    
            # Load and prepare image
            image = cv2.imread(str(file))
            creationTime = os.path.getmtime(file)
            dt_c = datetime.datetime.fromtimestamp(creationTime)
            formatted_datetime = dt_c.strftime("%m/%d/%Y %H:%M")
            datetimes.append(formatted_datetime)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w, *_ = image.shape
            
            # Apply cropping if specified
            if crop_params:
                y_start, y_end, x_start, x_end = crop_params
                image_for_model = image[y_start:y_end, x_start:x_end]
            else:
                image_for_model = image
            
            # Resize for model input
            image_for_model = cv2.resize(image_for_model, (224, 224))
            image_for_model = image_for_model / 255.0   

            # Get file info
            file_path = Path(file)
            filename = file_path.name
            Camera = file_path.stem.split('_')[0]
            
            # Prepare tensor for model
            image_tensor = np.transpose(image_for_model, (2, 0, 1))
            image_tensor = torch.tensor(image_tensor, dtype=torch.float)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)

            # Get predictions (in crop coordinate system if crop was applied)
            outputs = model(image_tensor)
            outputs = outputs.cpu().numpy() 
            pred_keypoint_crop = np.array(outputs[0], dtype='float32')

            # Transform keypoints back to original image coordinates
            pred_keypoint_orig = transform_keypoints_to_original(pred_keypoint_crop, crop_params, (orig_h, orig_w))

            # For visualization, use the processed image and crop coordinates
            if i % 100 == 0: 
                image_viz = image_tensor.squeeze().cpu()
                image_viz = np.transpose(image_viz, (1, 2, 0))
                image_viz = np.array(image_viz, dtype='float32')
                vis_predicted_keypoints(filename, image_viz, pred_keypoint_crop)
            
            # Extract coordinates
            x1_pred_crop, y1_pred_crop, x2_pred_crop, y2_pred_crop = pred_keypoint_crop[0], pred_keypoint_crop[1], pred_keypoint_crop[2], pred_keypoint_crop[3]
            x1_pred_orig, y1_pred_orig, x2_pred_orig, y2_pred_orig = pred_keypoint_orig[0], pred_keypoint_orig[1], pred_keypoint_orig[2], pred_keypoint_orig[3]
            
            # Store results
            Cameras.append(Camera)
            filenames.append(filename)
            
            # Store crop coordinates
            x1s_pred_crop.append(x1_pred_crop), y1s_pred_crop.append(y1_pred_crop), x2s_pred_crop.append(x2_pred_crop), y2s_pred_crop.append(y2_pred_crop)
            
            # Store original coordinates
            x1s_pred_orig.append(x1_pred_orig), y1s_pred_orig.append(y1_pred_orig), x2s_pred_orig.append(x2_pred_orig), y2s_pred_orig.append(y2_pred_orig)
            
            # Calculate measurements using ORIGINAL coordinates
            total_length_pixel = distance.euclidean([x1_pred_orig, y1_pred_orig], [x2_pred_orig, y2_pred_orig])
            total_length_pixels.append(total_length_pixel)

            ## snow depth conversion using original coordinates
            try: 
                full_length_pole_cm = metadata[metadata['camera_id'] == Camera]['pole_length_cm'].values[0]
                pixel_cm_conversion = metadata[metadata['camera_id'] == Camera]['pixel_cm_conversion'].values[0] 
                snow_depth = full_length_pole_cm - (pixel_cm_conversion * total_length_pixel)
                snow_depths.append(snow_depth)
            except Exception as e: 
                print(f"Error processing {Camera}: {e}")
                ## if you don't have a metadata stored properly it will just insert a 0 for snowdepth
                snow_depths.append(0)
            
    # Create results dataframe with both coordinate systems
    results = pd.DataFrame({
        'camera_id': Cameras, 'filename': filenames, 'datetime': datetimes,
        # Crop coordinates (224x224 space)
        'x1_pred_crop': x1s_pred_crop, 'y1_pred_crop': y1s_pred_crop, 
        'x2_pred_crop': x2s_pred_crop, 'y2_pred_crop': y2s_pred_crop,
        # Original coordinates (original image space)
        'x1_pred_orig': x1s_pred_orig, 'y1_pred_orig': y1s_pred_orig, 
        'x2_pred_orig': x2s_pred_orig, 'y2_pred_orig': y2s_pred_orig,
        # Measurements (based on original coordinates)
        'total_length_pixel': total_length_pixels, 'snow_depth': snow_depths
    })
    
    results.to_csv(f"{args.model}/predictions/results.csv")
    
    # Save crop information to results
    if crop_params:
        with open(f"{args.model}/predictions/crop_info.txt", "w") as f:
            f.write(f"Crop parameters used: y_start={crop_params[0]}, y_end={crop_params[1]}, x_start={crop_params[2]}, x_end={crop_params[3]}\n")
            f.write("Note: *_crop coordinates are in the 224x224 model input space\n")
            f.write("*_orig coordinates are in the original image coordinate system\n")
            f.write("Measurements (total_length_pixel, snow_depth) are based on original coordinates\n")
        print(f"Crop information saved to {args.model}/predictions/crop_info.txt")

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

    model = load_model(args)
    device = args.device  # Use the actual device from args instead of hardcoded 'cpu'
    results = predict(model, args, device)
    
    print(f"\nPrediction complete! Results saved to {args.model}/predictions/results.csv")
    print(f"Processed {len(results)} images")
    if len(results) > 0:
        print(f"Average snow depth: {results['snow_depth'].mean():.2f} cm")

if __name__ == '__main__':
    main()