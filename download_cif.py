import os
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import requests

def download_cif(cod_id):
    if not os.path.exists('./data'):
        os.mkdir('./data')
    cif_path = f'./data/{cod_id}.cif'
    if not os.path.isfile(cif_path):
        url = f"http://www.crystallography.net/cod/{cod_id}.cif"
        response = requests.get(url)
        if response.status_code == 200:
            with open(cif_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download {cod_id}.cif. Status code: {response.status_code}")

def read_file(file_path):
    with open(file_path, 'r') as file:
        cod_link = file.read().splitlines()

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(tqdm(
                executor.map(download_cif, cod_link), total = len(cod_link)
                ))

if __name__ == '__main__':
    read_file(sys.argv[1])