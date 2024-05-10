from pymatgen.io.cif import CifParser
import nglview as nv

# Parse the CIF file
parser = CifParser("data/9000000.cif")
structure = parser.get_structures()[0]

# Create a NGLview structure
view = nv.show_pymatgen(structure)

# Display the structure
view