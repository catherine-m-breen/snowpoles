"""
This script will download the folder with the model to a folder called "model". 
We will use the wget command to download the zip folder. 
"""

import os


def download_models():
    """
    see the Zenodo page for the latest models
    """
    root = os.getcwd()
    save_path = f"{root}/models"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    url = "https://zenodo.org/records/12764696/files/CO_and_WA_model.pth"

    # download if does not exist
    if not os.path.exists(f"{save_path}/CO_and_WA_model.pth"):
        wget_command = f"wget {url} -P {save_path}"
        os.system(wget_command)
        return print("\n models download! \n")
    else:
        return print("model already saved")


download_models()
