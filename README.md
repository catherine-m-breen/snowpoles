## overview
We present a model that faciltates snow depth extraction from snow poles by identifying the top and bottom of the pole and calculating the length of the pole in pixels. The model contains a neural network with ResNet50 architecture (pre-trained with ImageNet) trained on 9721 images of snowpoles installed in front of time-lapse cameras. The images are from 32 different sites in Okanogan County, Washington, USA, and Grand Mesa, Colorado, USA. We welcome testing of the model on your site, but we recommend an additional training step for best results in your area of interest. More info on each script in the codebase is below. 

### Example images (image: left; model prediction: right)

<img src="https://github.com/CV4EcologySchool/snow-Dayz/blob/main/snowpoles/example_imgs/E6A_WSCT0293.JPG" style="width: 350px;"> <img src="https://github.com/CV4EcologySchool/snow-Dayz/blob/main/snowpoles/example_imgs/eval_E6A_WSCT0293.JPG.png" width="50%">

> [!IMPORTANT]  
> Because the model was trained on data in Washington and Colorado, the accuracy on your data for the model "off the shelf" may be lower than what we reported in the paper. To obtain better accuracy, we recommend fine-tuning the model (more on that below). If you'd just like to try the model off the shelf, you can skip right to step 3 (Predictions), which will run the model as is on your data without fine-tuning. 

## Retraining for more accurate predictions

Our findings suggested that some labeling of the dataset of interest improved the performance of the model on new datasets. We recommend following the steps below to 1) label a subset of images from each camera from your study for best results. 2) Fine-tune the model using the subset. Then 3) predict on all of your data, and 4) convert to snow depth. We will first describe how to use the model for predictions, then we will explain how to re-train the model for best results. The overall workflow is summarized in the flowchart below. 

<img src="https://github.com/CV4EcologySchool/snow-Dayz/blob/main/snowpoles/example_imgs/flowchart3.png"> 


## 1. Labeling subset of images 

- We provide labeling.py to facilitate labeling. The labels are then saved in the right format for re-training the model. To label your own images run the following updated the arguments with your specific data paths and pole measurements. 
    - '--datapath' assumes that your original iamges are saved in a nested subfolder from the root folder called "data". Each camera folder has a unique folder ID that matches the camera ID.
    - '--savedir' will save the labels.csv in your data directory 
    - '--pole_length' height of your poles. If they are varying you will need to run this on each individual folder and then combine all the labels into one csv. 

```
python src/labeling.py --datapath '/Users/Documents/data' --savedir '/Users/Documents/data' --pole_length '304.8'
```

> [!NOTE]  
> It will also create a folder called 'train_data' that will serve as the training folder for model training. 


## 2. Fine-tuning model without GPU
- Before training, change the dataset root in the configuration files in `config` file. Then, train (model will automatically default to CPU if no cuda found). We simplified the training step, so that if you use the labeling script no other prep is needed except for updating the paths in the config file.

on local or GPU machine: 
```
python src/train.py
```


## 3. Predictions

> [!TIP]
> Start here if you don't want to fine-tune the model, but just want to try the model on your data.

- To test the model on your own sites of interest, run 'predict.py'. The script saves the results as a .csv as well as pictures of the predictions. On a local machine, the script can process about 1.1 image/ second. So, 1000 images would take ~18 min to run. The script contains four arguments to allow the user to customize predictions: 1) model_folder, 2) dir_path, 3) folder_path, and 4) output_path. 
    - 'model_path' if the user has retrained the model and would like to point it to the new folder, they can update the path here. If the user leaves this blank, it will automatically use the model developed in the corresponding paper which is saved in this directory. 
    - "dir_path" is the argument if you would like to predict  for a directory of camera folders. It assumes that original images are saved in a nested subfolder from a root folder, and that each camera folder has a unique folder ID that matches the camera ID. For example the directory may be called "data" and then each folder within "data" is "camera1," "camera2," etc, where within each camera folder are the .jpg images. This is the most efficient way to run the code, because you only have to run one line, and the model will make predictions across all your camera folders. It takes about XX to process. 
    - "folder_path" is similiar to "dir_path" except that it does not assume a nested folder structure. This can be used if all your images are in one folder, or you simply want to run predictions on one folder only. If left blank, it will default to example images. 
        **Note:** You only need to use dir_path or folder_path not both. 
    - 'output_path'is the folder where you would like to save the model predictions. The script will automatically create a subfolder called "predictions" where it will save the .csv of the top and bottom coordinates as well as the length in pixels. It will also save the pictures of the predictions in this folder as well. (mandatory) 

- An example for a directory of  from the command line is as follows: 
on local or GPU machine:

```
python src/predict.py --dir_path '/Users/Documents/data' --output_folder '/Users/Documents/data'
```

- An example for a single folder from the command line is as follows: 
on local or GPU machine:

```
python src/predict.py --folder_path '/Users/Documents/data/CAMERA1' --output_folder '/Users/Documents/data/CAMERA1'
```

**Note: the script only allows for a dir_path OR a folder_path because of how it uploads the images to local memory. Please use only one argument or the other. If both are provided, it will default to folder_path.

## 4. Snow Depth Extraction

- Once the model has predicted the top and bottom of the pole, it is time to convert to snow depth. The script is called 'depth_conversion.py'.The script contains two arguments to allow the user to customize the predictions path. 
    - 'predictions_path' the folder where the predictions are stored (created automatically from predict.py)

```
python src/depth_conversion.py --predictions_path '/Users/Documents/data/CAMERA1' 
```

## Basic packages:
- pytorch
- numpy


## other resources
Breen, C. M., C. Hiemstra, C. M. Vuyovich, and M. Mason. (2022). SnowEx20 Grand Mesa Snow Depth from Snow Pole Time-Lapse Imagery, Version 1 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/14EU7OLF051V. Date Accessed 04-13-2023.

