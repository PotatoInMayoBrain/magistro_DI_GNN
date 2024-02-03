import sys
from ase.io import read
from scipy.spatial import Voronoi
import numpy as np
import csv

# Perform Voronoi Tessellation
def perform_voronoi_tessellation(atomic_coordinates):
    vor = Voronoi(atomic_coordinates)
    return vor

# Determine Adjacency Relationships
def determine_adjacency_relationships(vor, num_atoms):
    adjacency_list = []
    for i, region in enumerate(vor.regions):
        if not -1 in region and len(region) > 0:
            for j in region:
                if j != i and j != -1 and j < num_atoms:
                    adjacency_list.append((i, j))
    return adjacency_list

# Create Adjacency Matrix
def create_adjacency_matrix(adjacency_list, num_atoms):
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    for i, j in adjacency_list:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    return adjacency_matrix

# Write Adjacency Matrix to stdout
def write_adjacency_matrix_to_stdout(adjacency_matrix):
    writer = csv.writer(sys.stdout)
    writer.writerows(adjacency_matrix)

if __name__ == "__main__":
    # Read CIF file from stdin
    cif_content = sys.stdin.readlines()

    # Parse CIF file
    structure = read(io.StringIO(''.join(cif_content)), format="cif")
    atomic_coordinates = structure.get_positions()

    # Perform Voronoi Tessellation
    vor = perform_voronoi_tessellation(atomic_coordinates)

    # Determine Adjacency Relationships
    adjacency_list = determine_adjacency_relationships(vor, len(atomic_coordinates))

    # Create Adjacency Matrix
    adjacency_matrix = create_adjacency_matrix(adjacency_list, len(atomic_coordinates))

    # Write Adjacency Matrix to stdout
    write_adjacency_matrix_to_stdout(adjacency_matrix)
