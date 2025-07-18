'''
Short code to combine the locations with the results. This is sometimes called a "join," where you 
combine two dataframes based on a "key" (like camera ID). 

In this case we have two dataframes, 1) the dataframe of your model outputs which has camera ID and filename 
as two separate columns and 2) the locations metadata that Morgan provided. 

We will conduct a "join" to combine the two dataframes based on the camera ID. Think about it this way: the code
will walk through our dataframe of model outputs and for each camera ID, it will look up the location for that 
respective camera ID in Morgan's table and add it as a column. 

Note: the reason why the locations weren't originally written into the script is because the locations are usually 
provided as a separate file anyways.

To run: 

from directory 
python add_location_data.py

'''

# library to do the join
import pandas as pd
import IPython
from pathlib import Path

###################### UPDATE WITH OWN PATH HERE ###############
data = pd.read_csv(Path("C:/Users/SnowE/Documents/snowpoles-main/snowpoles-main/predictions/results.csv"))

# helper function to split cameraIDs to site IDs if there is a year at the end. 
def split_string(s):
    if "_" in s: 
        return s.split("_")[0]
    else: 
        return s[:-4]

data['Site ID'] = [split_string(i) for i in data['camera_id']]

################## UPDATE WITH OWN PATH HERE #################
locations = pd.read_csv(Path("C:/Users/SnowE/Downloads/1854TA_SnowStation_Locations.csv"))

merged = pd.merge(data, locations, on = "Site ID")


###################### UPDATE WITH OWN SAVE PATH HERE ################
merged.to_csv('C:/Users/SnowE/Documents/snowpoles-main/snowpoles-main/predictions/results_WITHLOCATIONS.csv')
