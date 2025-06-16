'''
written by Catherine Breen
July 2024

run with:
python preprocess/rename_photos.py --path "./nontrained_data"

'''


import glob
import tqdm
import argparse
from pathlib import Path

def main():
    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description='Predict top and bottom coordinates.')
    parser.add_argument('--path', required=False, help = 'path to images to rename with camera ID in front', default = 'example_nontrained_data')
    args = parser.parse_args()


    files = list(Path(args.path).rglob("*"))  # recursively grab all files

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

if __name__ == '__main__':
    main()




