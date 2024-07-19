## overview
We present a model that faciltates snow depth extraction from snow poles by identifying the top and bottom of the pole and calculating the length of the pole in pixels that can be converted to depth. The model contains a neural network with ResNet50 architecture (pre-trained with ImageNet) trained on 9721 images of snowpoles installed in front of time-lapse cameras. The images are from 32 different sites in Okanogan County, Washington, USA, and Grand Mesa, Colorado, USA. We welcome testing of the model on your site, but we recommend an additional training step for best results in your area of interest. More info on each script in the codebase is below. 

### Example images (image: left; model prediction: right)

<img src="https://github.com/CV4EcologySchool/snow-Dayz/blob/main/snowpoles/example_imgs/E6A_WSCT0293.JPG" style="width: 350px;"> <img src="https://github.com/CV4EcologySchool/snow-Dayz/blob/main/snowpoles/example_imgs/eval_E6A_WSCT0293.JPG.png" width="50%">

## Model environment

To set up the model for the demo and all remaining scripts, we want to make sure we have the right Python environment. To do so, download the github folder, navigate to the folder in your command line, then run the following to install the appropriate packages for the model. 

```
conda env update -f environment.yml
```

## Model demo

We recommend starting with the demo script to ensure proper folder, model download, and conda environment set-up. For the demo, run the following line after navigating to this folder in your command line. The script is set-up to use the example files (from 'example_data') and to download the model to your local machine. Note: If you had trouble on the conda install, feel free to try the src/demo.py script, installing the packages listed in the script using your preferred enviroment manager. 

```
conda activate snowkeypoint
python src/demo.py 
```
This will create a folder called "demo_predictions" in your downloaded repository folder with the predictions from the model and a csv of snowdepth. It will also create a local folder on your machine with the trained model and weights. 

> [!IMPORTANT]  
> Because the model was trained on data in Washington and Colorado, the accuracy on your data for the model "off the shelf" may be lower than what we reported in the paper. To obtain better accuracy, we recommend fine-tuning the model (more on that below). If you'd like the model off the shelf, you can skip right to step 3 (Predictions), which will run the model as is on your data without fine-tuning. 

## Retraining for more accurate predictions

Our findings suggested that some labeling of the dataset of interest improved the performance of the model on new datasets. We recommend following the steps below to 1) label a subset of images from each camera from your study for best results. 2) Fine-tune the model using the subset. Then 3) predict on all of your data and convert to snow depth. The overall workflow is summarized in the flowchart below. 

<img src="https://github.com/CV4EcologySchool/snow-Dayz/blob/main/snowpoles/example_imgs/flowchart3.png"> 


## 1. Labeling subset of images 

- We provide labeling.py to facilitate labeling. The labels are then saved in the right format for re-training the model. To label your own images run the following, updating the arguments with your specific data paths and pole measurements. It will create .csv's called "labels" and "metadata." 'labels" will have the right information for the model, and 'metadata' has the information for the script to subsequently convert to snow depth. 
    - '--datapath' assumes that your original iamges are saved in a nested subfolder from the root folder called "data". Each camera folder has a unique folder ID that matches the camera ID.
    - '--pole_length' height of your poles. If they are varying you will need to run this on each individual folder and then combine all the labels into one csv.
    - 'subset_to_label' this is the spacing between images for each camera. We found that the model is more accurate with less space between images, but about every 10 was the right sequence
 
Note: our script "rename_photos.py" checks that the image filenames are in the right format and updates them if not. It assumes the images are in the following tree: image dir > image folder with camera id > filename (see nontrained_data). 

GENERIC:
```
python preprocess/rename_photos.py 
python src/labeling.py --datapath [IMAGE_DIRECTORY] --pole_length [POLE LENGTH IN CM] --subset_to_label [# BETWEEN LABELED IMAGES]
```
EXAMPLE:
```
python preprocess/rename_photos.py 
python src/labeling.py --datapath 'nontrained_data' --pole_length '304.8' --subset_to_label '10'
```

> [!NOTE]  
> The example 'nontrained_data' folder above uses data that was not included in the paper, from the NASA SnowEx 2017 campaign (Raliegh et al. 2022). We used these images to beta test this pipeline further. Simply delete the contents of this folder and replace with the camera folders with your images. We left the data in there so you can see how we organized the camera folders and what labeling.py will generate (labels.csv and metadata.csv)


## 2. Fine-tuning model without GPU
- Now it's time to train! Make sure to update the `config` file. Then, train (model will automatically default to CPU if no cuda found). We simplified the training step, so that if you use the labeling script no other prep is needed except for updating the paths in the config file.

on local or GPU machine: 
```
python src/train.py
```


## 3. Predictions

> [!TIP]
> Start here if you don't want to fine-tune the model, but just want to try the model on your data.

- To test the model on your own sites of interest, run 'predict.py'. The script saves the results as a .csv as well as pictures of the predictions. On a local machine, the script can process about 1.1 image/ second. So, 1000 images would take ~18 min to run. The script contains four arguments to allow the user to customize predictions: 1) model_folder, 2) dir_path, 3) folder_path, and 4) output_path. The script takes the following arguments: 
    - 'model_path' if the user has retrained the model and would like to point it to the new folder, they can update the path here. If the user leaves this blank, it will automatically use the model developed in the corresponding paper which is saved in this directory. 
    - "img_dir" is the argument if you would like to predict for a directory of camera folders. It assumes that original images are saved in a nested subfolder from a root folder, and that each camera folder has a unique folder ID that matches the camera ID. For example the directory may be called "data" and then each folder within "data" is "camera1," "camera2," etc, where within each camera folder are the .jpg images. This is the most efficient way to run the code, because you only have to run one line, and the model will make predictions across all your camera folders. 
    - "img_folder" is similiar to "dir_path" except that it does not assume a nested folder structure. This can be used if all your images are in one folder, or you simply want to run predictions on one folder only. If left blank, it will default to example images. 
        **Note:** You only need to use dir_path or folder_path not both. 
    - "metadata" this stores the information for the model to convert to snowdepth (automatically created if use labeling.py)

- An example for a directory of  from the command line is as follows: 
on local or GPU machine:

GENERIC: 
```
python src/predict.py --model_path [PATH TO CUSTOMIZED MODEL] --img_dir [IMAGE DIRECTORY]  --metadata [POLE METADATA]
```
EXAMPLE:
```
python src/predict.py --model_path './output1/model.pth' --img_dir './example_nontrained_data'  --metadata './example_nontrained_data/pole_metadata.csv'

```

- An example for a single folder from the command line is as follows: 
on local or GPU machine:

GENERIC: 
```
python src/predict.py --model_path [PATH TO CUSTOMIZED MODEL] --img_folder [IMAGE FOLDER]  --metadata [POLE METADATA]
```

EXAMPLE:
```
python src/predict.py --model_path './output1/model.pth' --img_folder './example_nontrained_data/TLS-A1N'  --metadata './example_nontrained_data/pole_metadata.csv'
```

**Note: the script only allows for a img_dir OR a img_folder because of how it uploads the images to local memory. Please use only one argument or the other. If both are provided, it will default to img_dir.

## 4. Snow Depth Extraction (optional)

- Predictions.py (see #3) will automatically convert to snow depth. However, if you find that you need to update some information (metadata, for example), you can run the script post-hoc. The script is called 'depth_conversion.py'.The script contains two arguments to allow the user to customize the predictions path. 
    - 'predictions_path' the predictions from the model (generated by predictions.py)
    - 'metadata' the pole_metadata.csv (generatd from labeling.csv)


GENERIC: 
```
python src/depth_conversion.py --predictions_path [PREDICTIONS CSV] --metadata [POLE METADATA CSV]
```
EXAMPLE:
```
python src/depth_conversion.py --predictions_path '/predictions/results.csv' --metadata 'example_nontrained_data/pole_metadata.csv'
```

## Basic packages:
- pytorch
- numpy


## other resources
Breen, C. M., C. Hiemstra, C. M. Vuyovich, and M. Mason. (2022). SnowEx20 Grand Mesa Snow Depth from Snow Pole Time-Lapse Imagery, Version 1 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/14EU7OLF051V. Date Accessed 04-13-2023.

Raleigh, M. S., W. R. Currier, J. D. Lundquist, P. Houser and C. Hiemstra. 2022. SnowEx17 TimeLapse Imagery, Version 1. [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/WYRNU50R9L5R. Date Accessed 04-14-2023.

