
'''
This script will download the folder with the model and the demo data to a location of your choice. 
We will use the wget command to download the zip folder. 
'''



import os
import datetime
import IPython

def wget():

    ## users have read access to the following folder
    folder = 'https://www.dropbox.com/s/ew2jket9lisdf4oor/example.zip'
    
    command = f'wget {folder}' 
    print(command)
    os.system(command)

wget()

