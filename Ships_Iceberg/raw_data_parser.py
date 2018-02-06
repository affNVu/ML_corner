import os
import json
from pprint import pprint
# Declare working directory
cwd = os.getcwd()

def read_json(dir_path = '/dataset/train.json'):
    file_path = cwd + dir_path  # Folder contain the Kaggle dataset

    # Read the input JSON file
    X = []
    with open(file_path) as data_file:    
        data = json.load(data_file)
        # Parse data
        #for i in range(len(data)):
        for i in range(1):
            obj = data[i]

            # Each obj is a json with: band_1, band_2, inc_angle, id, is_iceberg
            band_1 = obj["band_1"]
            band_2 = obj["band_2"]
            inc_angle = obj["inc_angle"]
            id = obj["id"]
            is_iceberg = obj["is_iceberg"]

            # Put into shape 
            x = [band_1, band_2, inc_angle, id, is_iceberg]
            X.append(x)
    return X


read_json()