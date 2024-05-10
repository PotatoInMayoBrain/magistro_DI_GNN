import os
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_cif(cod_id):
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.isfile(f'./data/{cod_id}.cif'):
        os.system(f'wget -q -P ./data/ https://www.crystallography.net/cod/{cod_id}.cif')

def read_file(file_path):
    with open(file_path, 'r') as file:
        cod_ids = file.read().splitlines()

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(tqdm(executor.map(download_cif, cod_ids), total = len(cod_ids)))


read_file(sys.argv[1])
