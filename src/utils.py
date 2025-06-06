import matplotlib.pyplot as plt
import numpy as np
import config
import IPython
import cv2
import argparse
import math
import pandas as pd
import glob
import PIL
from PIL import Image
from PIL import ExifTags


def valid_keypoints_plot(image, outputs, orig_keypoints, epoch):
    """
    This function plots the regressed (predicted) keypoints and the actual
    keypoints after each validation epoch for one image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()
    # just get a single datapoint from each batch
    img = image[0]  ## something snow in it ## halfway throught the dataset
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]
    img = np.array(img, dtype="float32")
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)

    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        if p == 0:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], "r.")  ## top
            plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], "b.")
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], "r.")  ## bottom
            plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], "b.")
    plt.savefig(f"{config.OUTPUT_PATH}/val_epoch_{epoch}.png")
    plt.close()


def dataset_keypoints_plot(data):
    """
    #  This function shows the image faces and keypoint plots that the model
    # will actually see. This is a good way to validate that our dataset is in
    # fact corrent and the faces align wiht the keypoint features. The plot
    # will be show just before training starts. Press `q` to quit the plot and
    # start training.
    """
    plt.figure(figsize=(10, 10))
    for i in range(9):
        sample = data[i]
        img = sample["image"]
        img = np.array(img, dtype="float32")  # /255
        # IPython.embed()
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        keypoints = sample["keypoints"]
        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1], "b.")
    plt.show()
    plt.close()


def eval_keypoints_plot(file, image, outputs, eval, orig_keypoints):
    """
    This function plots the regressed (predicted) keypoints and the actual
    keypoints after each validation epoch for one image in the batch.
    'eval' is the method to check the model, whether is the valid data (eval) or test data (test)
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    # IPython.embed()
    image = image.detach().cpu()
    image = image.squeeze(0)  ## drop the dimension because no longer need it for model
    outputs = outputs  # .detach().cpu().numpy()
    orig_keypoints = (
        orig_keypoints  # .detach().cpu().numpy()#orig_keypoints.detach().cpu().numpy()
    )
    # just get a single datapoint from each batch
    output_keypoint = outputs[0]
    img = np.array(image, dtype="float32")
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)

    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoints = orig_keypoints.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        if p == 0:
            plt.plot(orig_keypoints[p, 0], orig_keypoints[p, 1], "b.", markersize=20)
            plt.plot(
                output_keypoint[p, 0], output_keypoint[p, 1], "r.", markersize=20
            )  ## top
        else:
            plt.plot(orig_keypoints[p, 0], orig_keypoints[p, 1], "b.", markersize=20)
            plt.plot(
                output_keypoint[p, 0], output_keypoint[p, 1], "r.", markersize=20
            )  ## bottom
    plt.savefig(f"{config.OUTPUT_PATH}/{eval}/{eval}_{file}.png")
    plt.close()


def vis_keypoints(image, keypoints, color=(0, 255, 0), diameter=15):
    image = image.copy()

    for x, y in keypoints:
        print(x, y)
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.imshow(image)
    plt.show()
    plt.close()


def vis_predicted_keypoints(
    args, file, image, keypoints, color=(0, 255, 0), diameter=15
):
    output_keypoint = keypoints.reshape(-1, 2)

    plt.imshow(image)
    for p in range(output_keypoint.shape[0]):
        if p == 0:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], "r.")  ## top
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], "r.")  ## bottom
    plt.savefig(f"{args.output_path}/predictions/image_{file}.png")
    plt.close()


def camres(Camera):
    df = pd.read_csv(f"{config.native_res_path}")
    try:
        orig_w = df.loc[df["camID"] == Camera, "orig_w"].iloc[0]
        orig_h = df.loc[df["camID"] == Camera, "orig_h"].iloc[0]
    except:
        print("error")
    return orig_w, orig_h


def conversionDic(Camera):
    conversion_table = pd.read_csv(f"{config.snowfreetbl_path}")
    convDic = dict(zip(conversion_table["camera"], conversion_table["conversion"]))
    conversion = convDic[Camera]

    stake_cm_dic = dict(
        zip(conversion_table["camera"], conversion_table["snow_free_cm"])
    )  ## snowfree stake
    snowfreestake_cm = stake_cm_dic[Camera]

    return conversion, snowfreestake_cm


def outputs_in_cm(Camera, filename, x1s_pred, y1s_pred, x2s_pred, y2s_pred):
    """
    This function converts the length in pixels to length in cm for each output
    """
    orig_w, orig_h = camres(Camera)
    conversion, snowfreestake_cm = conversionDic(Camera)

    keypoints = [x1s_pred, y1s_pred, x2s_pred, y2s_pred]
    keypoints = np.array(keypoints, dtype="float32")
    keypoints = keypoints.reshape(-1, 2)
    keypoints = keypoints * [orig_w / 224, orig_h / 224]

    proj_pix_length = math.dist(keypoints[0], keypoints[1])
    proj_cm_length = proj_pix_length * float(conversion)
    snow_depth = snowfreestake_cm - float(proj_cm_length)
    x1_proj, y1_proj, x2_proj, y2_proj = (
        keypoints[0][0],
        keypoints[0][1],
        keypoints[1][0],
        keypoints[1][1],
    )

    cmresults = {
        "Camera": Camera,
        "filename": filename,
        "x1_proj": x1_proj,
        "y1_proj": y1_proj,
        "x2_proj": x2_proj,
        "y2_proj": y2_proj,
        "proj_pixel_length": proj_pix_length,
        "proj_cm_length": proj_cm_length,
        "snow_depth": snow_depth,
    }

    return cmresults


def datetimeExtrac(filename):
    datetimeinfo = pd.read_csv(f"{config.datetime_info}")
    fileDatetime = datetimeinfo.loc[
        datetimeinfo["filenames"] == filename, "datetimes"
    ].iloc[0]
    return fileDatetime


def diffcm(Camera, filename, automated_snow_depth):

    fileDatetime = datetimeExtrac(filename)
    actual_snow_depth = pd.read_csv(
        f"{config.manual_labels_path}"
    )  ## add CH and OK poles using conversions

    try:
        sd = float(
            actual_snow_depth[
                (actual_snow_depth["camera"] == Camera)
                & (actual_snow_depth["dates"] == fileDatetime)
            ]["snowDepth"]
        )
        manual_snowdepth = sd
        difference = manual_snowdepth - automated_snow_depth
    except:
        manual_snowdepth = "na"
        difference = "na"

    return manual_snowdepth, difference


def MAPE(Y_actual, Y_Predicted):
    mape = (np.abs(Y_actual - Y_Predicted) / Y_actual) * 100
    return mape
