import csv
from ase.io import read
from scipy.spatial import Voronoi
import numpy as np

def get_adjacency_matrix(structure):
    # Get atomic positions
    positions = structure.get_positions()

    # Compute Voronoi tessellation
    vor = Voronoi(positions)

    # Initialize adjacency matrix
    n_atoms = len(positions)
    adjacency_matrix = np.zeros((n_atoms, n_atoms), dtype=int)

    # Iterate over Voronoi vertices
    for region in vor.regions:
        if -1 not in region and len(region) > 0:
            # Get atom indices forming the Voronoi cell
            atoms_in_cell = [vor.point_region[i] for i in region if i < len(vor.point_region) and vor.point_region[i] < n_atoms]

            # Update adjacency matrix
            for i in range(len(atoms_in_cell)):
                for j in range(i+1, len(atoms_in_cell)):
                    atom1 = atoms_in_cell[i]
                    atom2 = atoms_in_cell[j]
                    adjacency_matrix[atom1][atom2] = 1
                    adjacency_matrix[atom2][atom1] = 1

    return adjacency_matrix

# Path to the CIF file
cif_file = "4303837.cif"

# Read the structure from the CIF file
structure = read(cif_file)

# Generate the adjacency matrix
adj_matrix = get_adjacency_matrix(structure)

# Write adjacency matrix to a CSV file
csv_file = "adjacency_matrix.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(adj_matrix)

print(f"Adjacency matrix has been saved to {csv_file}")
