'''
written by Catherine Breen
July 2024

run with:
python preprocess/rename_photos.py --path 'example_nontrained_data'

'''


import glob
import os 
import tqdm
import argparse
import IPython

def main():
    # Argument parser for command-line arguments:
    parser = argparse.ArgumentParser(description='Predict top and bottom coordinates.')
    parser.add_argument('--path', required=False, help = 'path to images to rename with camera ID in front', default = 'example_nontrained_data')
    args = parser.parse_args()

    files = glob.glob(f'{args.path}/**/*')
    for file in tqdm.tqdm(files): 
        try:
            camera = file.split('/')[-2]
            filename = file.split('/')[-1]
            ## check if already renamed: 
            if filename.startswith(f"{camera}_"): pass
            else: 
                new_filename = camera + '_' + filename
                os.rename(f'./{args.path}/{camera}/{filename}', f'./{args.path}/{camera}/{new_filename}')
        except Exception as e:
            print(f"{e} {file}")
            pass

if __name__ == '__main__':
    main()




