import os
import requests
import random
import pandas as pd
import tempfile
from CifFile import ReadCif
from tqdm import tqdm

with open ('COD-selection.txt', 'r') as f:
    links = f.read().splitlines()

# Randomly select 2000 links
random.shuffle(links)
links = links[:10]

data = []

# Iterate over the links and download each .cif file
for link in tqdm(links, desc='Downloading .cif files'):
    response = requests.get(link)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
        f.write(response.text)
        f.seek(0)  # Go back to the start of the file
        
        # Read the .cif file
        cif = ReadCif(f.name)

        # Get the data block
        data_block = list(cif.keys())[0]
        block = cif[data_block]

        # Extract properties
        atomic_positions = [block.get(f'_atom_site_fract_{dim}', None) for dim in ['x', 'y', 'z']]
        atomic_numbers = block.get('_atom_site_type_symbol', None)
        unit_cell_parameters = [block.get(f'_cell_length_{dim}', None) for dim in ['a', 'b', 'c']] + \
            [block.get(f'_cell_angle_{angle}', None) for angle in ['alpha', 'beta', 'gamma']]

        # Append the extracted data
        data.append([atomic_positions, atomic_numbers, unit_cell_parameters])

# Use pandas to process and store the data
df = pd.DataFrame(data, columns=['Atomic Positions', 'Atomic Numbers', 'Unit Cell Parameters'])
print(df.head())
df.to_csv('data.csv', index=False)