import sys
import pandas as pd
from pymatgen.io.cif import CifParser
from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import csv
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore")

file_exists = os.path.isfile('output.csv')

#Sukuriame funkciją, .cif failų nuskaitymui
def read_file(cif_file):
    file_path = os.path.join('./data', cif_file) # Sukuriame kintamąjį, kuriame saugomas .cif failo kelias
    try:
        with open(file_path, 'r') as file: # Atidarome .cif failą
            parser = CifParser(file_path) # Nuskaitome .cif failą
    except (AssertionError, ValueError) as e: # Jeigu .cif failas yra sugadintas, išvedame pranešimą ir grąžiname None
        print(f"Invalid or corrupted CIF file: {cif_file} error: {e}")
        return None  # 
    try:
        structure = parser.parse_structures(primitive=True)[0]
    except ValueError:
        print(f"No structures found in {cif_file}")
        return None  # or however you want to handle this case
    lattice = structure.lattice
    formula = structure.composition.reduced_formula
    volume = structure.volume
    #coordinates = structure.cart_coords
    #distances = pdist(coordinates)
    distances = structure.distance_matrix
    #distances = distances[np.triu_indices(distances.shape[0], k=1)]  # Convert to a 1D array
    try: 
        space_group_symbol, space_group_number = structure.get_space_group_info()
    except TypeError:
        print(f"Could not determine space group for {cif_file}")
        space_group_symbol, space_group_number = None, None
        return None
    return {
        'codid': cif_file.split('.')[0],  # Assuming the codid is the filename without the .cif extension
        'sg_number': space_group_number,
        'reduced_formula': formula,
        'lattice_angle_alpha': lattice.angles[0],
        'lattice_angle_beta': lattice.angles[1],
        'lattice_angle_gamma': lattice.angles[2],
        'lattice_param_a': lattice.a,
        'lattice_param_b': lattice.b,
        'lattice_param_c': lattice.c,
        'volume': volume,
        'distance_matrix': distances  # Convert the numpy array to a list
    } 

cif_files = None
cif_files = [f for f in os.listdir('./data') if f.endswith('.cif')]

with open('output.csv', 'w', newline='') as csvfile, h5py.File('output.hdf5', 'a') as hdf5file:
    fieldnames = ['codid', 'sg_number', 'reduced_formula', 'lattice_angle_alpha', 'lattice_angle_beta', 'lattice_angle_gamma', 'lattice_param_a', 'lattice_param_b', 'lattice_param_c', 'volume']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    with ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(read_file, cif_files), total=len(cif_files)):
            if result is not None:
                row = {key: value for key, value in result.items() if key != 'distance_matrix'}
                writer.writerow(row)
                hdf5file.create_dataset(result['codid'], data=result['distance_matrix'])



#with ThreadPoolExecutor() as executor:
#    results = list(tqdm(executor.map(read_file, cif_files), total=len(cif_files)))
#    count = results + count
#print("Count: ", count)

#for i in tqdm(range(len(cif_files))):
#    count.append(read_file(cif_files[i], i))
#print(len(count))

#
## Get the CIF file name from the command line arguments
#cif_file = sys.argv[1]
#
## Extract the COD ID from the file name
#cod_id = os.path.splitext(os.path.basename(cif_file))[0]
#
## Parse the CIF file
#parser = CifParser(cif_file)
#
## Get the first structure from the CIF file
#structure = parser.parse_structures()[0]
##print(structure)
#
## Extract data
#lattice = structure.lattice
##species = [str(s) for s in structure.species]
#space_group_symbol, space_group_number = structure.get_space_group_info()
#coordinates = structure.cart_coords
#
## Calculate the pairwise distances between the atoms
#distances = pdist(coordinates)
#
## Convert the distances to a square matrix
#adjacency_matrix = squareform(distances)
## Remove the upper triangle of the matrix
#adjacency_matrix = np.tril(adjacency_matrix).tolist()[1:]
## Remove the zeros from the adjacency matrix
#for i in range(len(adjacency_matrix)):   
#    non_zero_indices = [j for j, element in enumerate(adjacency_matrix[i]) if element != 0]
#    adjacency_matrix[i] = [adjacency_matrix[i][j] for j in non_zero_indices]
#
## Create a DataFrame
#df = pd.DataFrame({ 
#    'cod_id': [cod_id],
#    'sg': [space_group_number],
#    'lattice_a': [lattice.a],
#    'lattice_b': [lattice.b],
#    'lattice_c': [lattice.c],
#    'lattice_alpha': [lattice.alpha],
#    'lattice_beta': [lattice.beta],
#    'lattice_gamma': [lattice.gamma],
#    #'species': [species],
#    'adjacency matrix': [adjacency_matrix],
#})
#
#df.dropna()
## Write the DataFrame to a CSV file
## Replace 'output.csv' with the desired output file name
#df.to_csv('output.csv', index=False, mode='a', header=not file_exists)
#
##plt.figure(figsize=(10, 10))
##sns.heatmap(adjacency_matrix, square=True, cmap='viridis')
##
### Add a title to the heatmap
##plt.title('Adjacency Matrix Heatmap')
##
### Show the heatmap
##plt.show()