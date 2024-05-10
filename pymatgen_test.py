import sys
import pandas as pd
from pymatgen.io.cif import CifParser
from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore")

file_exists = os.path.isfile('output.csv')

#Sukuriame funkciją, .cif failų nuskaitymui
def read_file(cif_file):
    file_path = os.path.join('./data', cif_file)
    try:
        with open(file_path, 'r') as file:
            parser = CifParser(file_path)
    except (AssertionError, ValueError) as e:
        print(f"Invalid or corrupted CIF file: {cif_file} error: {e}")
        return 0  # or however you want to handle this case
    try:
        structure = parser.parse_structures()[0]
    except ValueError:
        print(f"No structures found in {cif_file}")
        return 0  # or however you want to handle this case
    lattice = structure.lattice
    formula = structure.composition.reduced_formula
    volume = structure.volume
    #coordinates = structure.cart_coords
    #distances = pdist(coordinates)
    distances = structure.distance_matrix
    try: 
        space_group_symbol, space_group_number = structure.get_space_group_info()
    except TypeError:
        print(f"Could not determine space group for {cif_file}")
        space_group_symbol, space_group_number = None, None
        return 0
    return 1

count = 0
cif_files = None
cif_files = [f for f in os.listdir('./data') if f.endswith('.cif')]

with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(read_file, cif_files), total=len(cif_files)))
    count = results + count
print("Count: ", count)

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