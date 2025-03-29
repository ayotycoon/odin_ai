import itertools
import os
import shutil
from itertools import permutations
import random

import pandas as pd

from main.utils.index import get_all_files_recursive
from main.utils.path_utils import safe_access_path


output_folder_path = ".temp/data/csv"

# Remove folder if it exists
if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)

# Recreate the folder
os.makedirs(output_folder_path)


def gen_syntetic():

    input_folder_path = "data/csv/synthetic/brands"
    _output_folder_path = safe_access_path(f"{output_folder_path}/generated")



    us_cities = pd.read_csv("data/csv/synthetic/cityState/uscities.csv", header=0 )
    def get_random_us_cities_state(column = 'state_name_id'):
        return us_cities[column].sample(n=1).values.flatten()[0]

    sample_addr = pd.read_csv("data/csv/synthetic/cityState/sample_addr.csv", header=0 )['addr']
    def get_random_us_address():
        return sample_addr.sample(n=1).values.flatten()[0]

    files = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]
    for i, file in enumerate(files):
        headers = ["Description", "Category", "Sub_Category"]
        all_file_pd = pd.DataFrame(columns=headers)

        file_path = os.path.join(input_folder_path, file)
        print(f"Importing dataset: {file_path}")
        # Read the first file with headers, subsequent ones without headers
        input_df = pd.read_csv(file_path, header=0 )
        for index, row in input_df.iterrows():

            if 'Brand' not in row:
                continue
            brand = row['Brand']
            cat = row['Category']
            sub = row['Sub_Category']

            x0 =get_random_us_cities_state()
            x1 =get_random_us_cities_state('city')
            x2 =get_random_us_cities_state('state_id')
            x3 =get_random_us_cities_state('state_name')
            x4 = get_random_us_address()
            x5 = get_random_us_address() + ' '+ x2
            x6 = str(random.randrange(1000000, 999999999)) + ' '+get_random_us_cities_state('state_id')


            # arr = [brand ,get_random_us_address() , get_random_us_cities_state()]
            # combinations = list(itertools.permutations(arr))
            #
            # permutations =  [' '.join(combination) for combination in combinations]

            for x in [x0,x1,x2,x3,x4,x5, x6]:
                all_file_pd.loc[len(all_file_pd)] = [f"{brand} {x}",cat,sub]
        _output_folder_path_file = f"{_output_folder_path}/{random.randrange(9, 999999999)}_gen.csv"
        all_file_pd.to_csv(_output_folder_path_file)



def gen_copy():
    input_folder_path = "data/csv/copy"

    safe_access_path(output_folder_path)
    for x in get_all_files_recursive(input_folder_path):
        shutil.copy(x, f"{output_folder_path}/{x.replace(input_folder_path,'')}")

def instantiate_data():
    gen_syntetic()
    gen_copy()
if __name__ == '__main__':
    instantiate_data()
