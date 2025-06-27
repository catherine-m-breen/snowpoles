"""
written by Catherine Breen
July 2024

run with:
python preprocess/rename_photos.py --path "./nontrained_data"

"""

import glob
import tqdm
import argparse
from pathlib import Path
import tomllib
import os


def main():
    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(
        description='Rename images with their camera IDs. Paths to the images should be something like "Root/Camera1/Image1.jpg".'
    )
    parser.add_argument(
        "--path", required=False, help="directory where images are located, by camera"
    )
    parser.add_argument(
        "--no_confirm", required=False, help="skip confirmation", action="store_true"
    )
    args = parser.parse_args()

    # Get arguments from config file if they weren't specified
    with open("config.toml", "rb") as configfile:
        config = tomllib.load(configfile)
    if not args.path:
        args.path = config["paths"]["input_images"]

    # Confirmation
    if not args.no_confirm:
        print(
            "\n\n# The following options were specified in config.toml or as arguments:\n"
        )
        if (args.path.startswith("/") or args.path[1] == ":"):
            print(
                "Directory where images are located, by camera:\n"
                + str(args.path)
                + "\n"
            )
        else:
            print(
                "Directory where images are located, by camera:\n"
                + os.getcwd()
                + "/"
                + str(args.path)
                + "\n"
            )
        confirmation = str(input("\n\nIs this OK? (y/n) "))
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
    rename_photos(args.path)

def rename_photos(path):
    files = list(Path(path).rglob("*"))  # recursively grab all files

    for file in tqdm.tqdm(files):
        try:
            if file.is_file():
                camera = file.parent.name
                filename = file.name

                # Check if already renamed
                if filename.startswith(f"{camera}_"):
                    continue
                new_filename = f"{camera}_{filename}"
                new_path = file.parent / new_filename

                file.rename(new_path)

        except Exception as e:
            print(f"{e} {file}")
            continue

if __name__ == "__main__":
    main()
