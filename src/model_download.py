import argparse
import os
import subprocess
import tomllib

# Argument parser for command-line arguments:
def main():
    parser = argparse.ArgumentParser(description="Download model")
    parser.add_argument("--output", help="path where model should be saved")
    args = parser.parse_args()

    # Get arguments from config file if they weren't specified
    with open("config.toml", "rb") as configfile:
        config = tomllib.load(configfile)
    if not args.output:
        args.output = config["paths"]["trainee_model"]

    download_models()

def download_models(save_path="./models", save_name="CO_and_WA_model.pth", confirm=True):
    # see the Zenodo page for the latest models
    root = os.getcwd()
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    url = "https://zenodo.org/records/12764696/files/CO_and_WA_model.pth"

    # download if model does not exist
    if not os.path.exists(f"{save_path + "/" + save_name}"):
        save_path = save_path.replace("\\", "/")
        if confirm:
            curl_command = f'curl -I --ssl-no-revoke "{url}"'
            headers = subprocess.run(
                ["curl", "-s", "-I", "--ssl-no-revoke", url], stdout=subprocess.PIPE
            ).stdout.decode("utf-8")
            for header in headers.split("\n"):
                if header.startswith("content-length"):
                    size = header[header.find(": ") + 2 : -1] + ".00"
                    suffix = 0
                    suffixes = " KMGTPEZYRQ"
                    while len(size) > 6:
                        size = size[:-6] + "." + size[-6:-4]
                        suffix += 1
                    print("\n\nModel download size:", size, suffixes[suffix] + "B")
                    confirmation = str(input("\nIs this OK? (y/n) "))
                    if confirmation.lower() != "y":
                        if confirmation.lower() == "n":
                            print(
                                "\nEdit the config file, located at",
                                os.getcwd()
                                + "/config.toml, to your liking, and then re-run this file.\n",
                            )
                        else:
                            print("Invalid input.\n")
                        quit()
        print("\nDownloading model...")
        curl_command = f'curl -L --ssl-no-revoke "{url}" -o "{save_path + "/" + save_name}"'
        os.system(curl_command)
    else:
        return print("\nA file with the specified output name already exists:\n" + save_path + "/" + save_name, "\n\nCheck config.toml if you did not specify a file on the command line.")


if __name__ == "__main__":
    main()
