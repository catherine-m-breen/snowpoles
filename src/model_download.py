import os

# Download model to models/ directory
def download_models():
    """
    see the Zenodo page for the latest models
    """
    root = os.getcwd()
    save_path = f"{root}/models" ## switched this
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    url = "https://zenodo.org/records/12764696/files/CO_and_WA_model.pth"

    # download if does not exist
    # on windows!! make sure 
    if not os.path.exists(f"{save_path}\CO_and_WA_model.pth"):
    
        ## wget_command = f"wget {url} -P {save_path}" ## this is linux
        save_path = save_path.replace("\\", "/")
        output_file = os.path.join(save_path, url.split("/")[-1]).replace("\\", "/")
        curl_command = f'curl -L --ssl-no-revoke "{url}" -o "{output_file}"'
        print(curl_command)
        os.system(curl_command)
        return print("\n models download! \n")
    else:
        return print("model already saved")

if __name__ == "__main__":
    download_models()
