from ase.io import read
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import csv

# Step 1: Parse CIF File
def parse_cif_file(cif_file):
    structure = read(cif_file)
    return structure

# Step 2: Perform Voronoi Tessellation
def perform_voronoi_tessellation(structure):
    atomic_coordinates = structure.get_positions()
    vor = Voronoi(atomic_coordinates)
    return vor

# Step 3: Determine Adjacency Relationships
def determine_adjacency_relationships(vor):
    # Perform analysis on Voronoi diagram to determine adjacency relationships
    # For example, you might consider neighboring Voronoi cells as adjacent atoms
    # You can customize this logic based on your specific requirements
    adjacency_list = []
    # Your logic to determine adjacency relationships
    return adjacency_list

# Step 4: Create Adjacency Matrix
def create_adjacency_matrix(adjacency_list, num_atoms):
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    for i, j in adjacency_list:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    return adjacency_matrix

# Step 5: Write to File
def write_adjacency_matrix_to_csv(adjacency_matrix, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(adjacency_matrix)

if __name__ == "__main__":
    cif_file = '4303837.cif'
    output_file = '4303837.csv'

    # Step 1: Parse CIF File
    structure = parse_cif_file(cif_file)

    # Step 2: Perform Voronoi Tessellation
    vor = perform_voronoi_tessellation(structure)

    # Step 3: Determine Adjacency Relationships
    adjacency_list = determine_adjacency_relationships(vor)

    # Step 4: Create Adjacency Matrix
    num_atoms = len(structure)
    adjacency_matrix = create_adjacency_matrix(adjacency_list, num_atoms)

    # Step 5: Write to File
    write_adjacency_matrix_to_csv(adjacency_matrix, output_file)
