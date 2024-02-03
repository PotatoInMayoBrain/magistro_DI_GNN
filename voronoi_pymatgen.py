from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN
import numpy as np

def get_adjacency_matrix(structure):
    voronoi = VoronoiNN(targets=[1])  # Considering only nearest neighbors
    adjacency_matrix = np.zeros((len(structure), len(structure)), dtype=int)
    
    atoms_with_no_neighbors = []  # Store indices of atoms with no neighbors
    
    for i in range(len(structure)):
        neighbors = voronoi.get_nn_info(structure, i)
        if not neighbors:
            atoms_with_no_neighbors.append(i)
            continue
        for neighbor in neighbors:
            j = structure.index(neighbor['site'])
            adjacency_matrix[i][j] = 1
    
    return adjacency_matrix, atoms_with_no_neighbors

# Path to the CIF file
cif_file = "4303837.cif"

# Read the structure from the CIF file
structure = Structure.from_file(cif_file)

# Generate the adjacency matrix and identify atoms with no neighbors
adj_matrix, no_neighbor_atoms = get_adjacency_matrix(structure)

print("Adjacency Matrix:")
print(adj_matrix)

if no_neighbor_atoms:
    print("Atoms with no neighbors:")
    print(no_neighbor_atoms)
else:
    print("All atoms have neighbors.")
