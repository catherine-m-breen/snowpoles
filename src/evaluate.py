"""

load model and run on data points 
export the csv of the data points and just use the bottom

example command line to run:

(make sure config file is set to the right model!)
python src/evaluate.py

"""

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
    model = snowPoleResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    checkpoint = torch.load(config.OUTPUT_PATH + "/model.pth")
    print(f"loading model from the following path: {config.OUTPUT_PATH}")
    # load model weights state_dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(model, data, eval="eval"):

    if not os.path.exists(f"{config.OUTPUT_PATH}/{eval}"):
        os.makedirs(f"{config.OUTPUT_PATH}/{eval}", exist_ok=True)

    output_list = []
    Cameras, filenames = [], []
    x1s_true, y1s_true, x2s_true, y2s_true = [], [], [], []
    x1s_pred, y1s_pred, x2s_pred, y2s_pred = [], [], [], []
    top_pixel_errors, bottom_pixel_errors, total_length_pixels = [], [], []
    total_length_pixel_actuals = []
    mape_errors = []
    mape_errors_sd = []
    mape_errors_sd_clean = []

    automated_sds, manual_sds, diff_sds = [], [], []

    metadata = pd.read_csv(f"{config.metadata}")
    labels = pd.read_csv(f"{config.labels}")

    with torch.no_grad():
        for i, data in tqdm(enumerate(data)):
            image, keypoints = data["image"].to(config.DEVICE), data["keypoints"].to(
                config.DEVICE
            )
            filename = data["filename"]
            Camera = filename.split("_")[0]

            # flatten the keypoints
            keypoints = keypoints.detach().cpu().numpy().reshape(-1, 2)
            x1_true, y1_true, x2_true, y2_true = (
                keypoints[0, 0],
                keypoints[0, 1],
                keypoints[1, 0],
                keypoints[1, 1],
            )
            ## add an empty dimension for sample size
            image = image.unsqueeze(0)
            outputs = model(image)
            outputs = outputs.detach().cpu().numpy()

            utils.eval_keypoints_plot(
                filename, image, outputs, eval, orig_keypoints=keypoints
            )  ## visualize points
            pred_keypoint = np.array(outputs[0], dtype="float32")
            x1_pred, y1_pred, x2_pred, y2_pred = (
                pred_keypoint[0],
                pred_keypoint[1],
                pred_keypoint[2],
                pred_keypoint[3],
            )

            Cameras.append(Camera)
            filenames.append(filename)
            x1s_true.append(x1_true), y1s_true.append(y1_true), x2s_true.append(
                x2_true
            ), y2s_true.append(y2_true)
            x1s_pred.append(x1_pred), y1s_pred.append(y1_pred), x2s_pred.append(
                x2_pred
            ), y2s_pred.append(y2_pred)

            ## outputs proj and in cm
            total_length_pixel = distance.euclidean(
                [x1_pred, y1_pred], [x2_pred, y2_pred]
            )
            full_length_pole_cm = metadata[metadata["camera_id"] == Camera][
                "pole_length_cm"
            ].values[0]
            pixel_cm_conversion = metadata[metadata["camera_id"] == Camera][
                "pixel_cm_conversion"
            ].values[0]
            automated_sd = full_length_pole_cm - (
                pixel_cm_conversion * total_length_pixel
            )

            automated_sds.append(automated_sd)

            # ## difference between automated and manual
            manual_pixel_length = labels[labels["filename"] == filename][
                "PixelLengths"
            ].values[0]
            manual_snowdepth = full_length_pole_cm - (
                pixel_cm_conversion * manual_pixel_length
            )
            difference = manual_snowdepth - automated_sd
            manual_sds.append(manual_snowdepth), diff_sds.append(difference)

            ## error
            top_pixel_error = distance.euclidean([x1_true, y1_true], [x1_pred, y1_pred])
            bottom_pixel_error = distance.euclidean(
                [x2_true, y2_true], [x2_pred, y2_pred]
            )
            total_length_pixel = distance.euclidean(
                [x1_pred, y1_pred], [x2_pred, y2_pred]
            )
            total_length_pixel_actual = distance.euclidean(
                [x1_true, y1_true], [x2_true, y2_true]
            )

            # MAPE
            mape_error = utils.MAPE(total_length_pixel_actual, total_length_pixel)
            mape_error_sd = utils.MAPE(manual_snowdepth, automated_sd)
            mape_errors_sd.append(mape_error_sd)

            top_pixel_errors.append(top_pixel_error), bottom_pixel_errors.append(
                bottom_pixel_error
            ), total_length_pixels.append(total_length_pixel)
            total_length_pixel_actuals.append(
                total_length_pixel_actual
            ), mape_errors.append(mape_error)

    results = pd.DataFrame(
        {
            "Camera": Cameras,
            "filename": filenames,
            "x1_true": x1s_true,
            "y1_true": y1s_true,
            "x2_true": x2s_true,
            "y2_true": y2s_true,
            "x1_pred": x1s_pred,
            "y1s_pred": y1s_pred,
            "x2_pred": x2s_pred,
            "y2_pred": y2s_pred,
            "top_pixel_error": top_pixel_errors,
            "bottom_pixel_error": bottom_pixel_errors,
            "total_length_pixel": total_length_pixels,
            "total_length_pixel_actual": total_length_pixel_actuals,
            "automated_depth": automated_sds,
            "manual_snowdepth": manual_sds,
            "difference": diff_sds,
            "mape": mape_errors,
            "mape_sd": mape_errors_sd,
        }
    )

    #### overall average
    print("Overall Top Pixel Error")
    print(f"{np.mean(top_pixel_errors)} +/- {np.std(top_pixel_errors)} \n")
    print("Overall Bottom Pixel Error")
    print(f"{np.mean(bottom_pixel_errors)} +/- {np.std(bottom_pixel_errors)} \n")
    print(f"Mean Average Percent Error (MAPE):")
    print(f"{np.mean(mape_errors)} +/- {np.std(mape_errors)} \n")
    print("Overall difference in cm")
    print(f"{np.mean(diff_sds)} +/- {np.std(diff_sds)} \n")
    print("Overall difference in MAPE")
    print(f"{np.mean(mape_errors_sd)} +/- {np.std(mape_errors_sd)} \n")
    print("\n")

    results.to_csv(f"{config.OUTPUT_PATH}/{eval}/evaluation_results.csv")
    return results


def main():
    model = load_model()
    print("results on valid data\n")
    outputs = predict(model, valid_data, eval="wa")


if __name__ == "__main__":
    main()
