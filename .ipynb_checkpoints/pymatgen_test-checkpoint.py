from pymatgen import core
import numpy as np
import sys
import nglview as nv

#input = sys.argv[1]
#output = sys.argv[2]
input = "data/8107719.cif"
# Load the structure from the CIF file
structure = core.Structure.from_file(input)

view = nv.show_pymatgen(structure)

view
'''
# Print the formula of the structure
formula = structure.formula
print("Formula:", formula)

# Print the space group of the structure
space_group = structure.get_space_group_info()
print("Space group:", space_group)

#print("\n Distance matrix: ", structure.distance_matrix)

# Get the indices of the lower triangular part
indices = np.tril_indices(structure.distance_matrix.shape[0])

# Use the indices to get the lower triangular part
lower_triangular = structure.distance_matrix[indices]
# Split the lower triangular part into chunks at zeros
chunks = np.split(lower_triangular, np.where(lower_triangular == 0)[0])

#print(chunks[0:10])

# Convert each chunk to a string and join them with commas
chunks_str = ",".join(np.array2string(chunk[chunk != 0], separator = ',') for chunk in chunks[1:])

print(chunks_str)

# Save the lower triangular part to a CSV file in a single line
with open(output, 'a') as f:
    # Write the space group before distance matrix lower triangular part
    f.write(f"{formula},{space_group[1]},{[chunks_str]}")
'''