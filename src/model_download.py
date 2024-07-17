
'''
This script will download the folder with the model and the demo data to a location of your choice. 
We will use the wget command to download the zip folder. 
'''



import os
import datetime
import IPython

def wget():

    ## users have read and download access to the following folder
    ## It contains demo data and model
    folder = 'https://www.dropbox.com/scl/fo/9go1g2mm6fqr8sdqem7pb/ADsRpJb3LQ8uMY_rju4sni0?rlkey=4mn5rf1c3qjjqv0biu4bl6wp4&st=4lfakejc&dl=0'
    
    command = f'wget {folder}' 
    print(command)
    os.system(command)

wget()
