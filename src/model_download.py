import os


def download_models(save_path="./models", save_name="CO_and_WA_model.pth"):
    # see the Zenodo page for the latest models
    root = os.getcwd()
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    url = "https://zenodo.org/records/12764696/files/CO_and_WA_model.pth"

    # download if model does not exist
    if not os.path.exists(f"{save_path}/{save_name}"):
        save_path = save_path.replace("\\", "/")
        output_file = os.path.join(save_path, save_name).replace("\\", "/")
        curl_command = f'curl -L --ssl-no-revoke "{url}" -o "{output_file}"'
        print(curl_command)
        os.system(curl_command)
        return print("\n models download! \n")
    else:
        return print("model already saved")


if __name__ == "__main__":
    download_models()
