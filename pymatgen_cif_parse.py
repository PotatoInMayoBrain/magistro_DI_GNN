from concurrent.futures import ThreadPoolExecutor
import csv
import h5py
import os
from pymatgen.io.cif import CifParser
from tqdm import tqdm
from io import StringIO
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")

'''Šis kodas nuskaito .cif failus iš nurodyto katalogo (./data), talpina juos 
atmintyje, skaito jų turinį naudojant pymatgen biblioteką ir išsaugo norimus 
duomenis į .csv failą (eilutėje vienas kristalas, pagrindinė informacija),  
.hdf5 failą (artumo matrica) ir dar vieną .hdf5 failą (elementai kristale).'''

def cache_cif_files(directory):
    cif_files = os.listdir(directory)
    cif_files_content = {}
    for cif_file in tqdm(cif_files, desc="Caching CIF files"):
        if cif_file.endswith('.cif'):
            with open(os.path.join(directory, cif_file), 'r') as file:
                cif_files_content[cif_file] = file.read()
    return cif_files_content

def read_file(cif_file_name, cif_content):
    try:
        parser = CifParser(StringIO(cif_content))
        structure = parser.parse_structures(primitive=False)[0]
        
        if structure.num_sites > 100:
            print(f"Skipping {cif_file_name}: more than 100 atoms")
            return None, None
        
        elements = structure.composition.elements
        valence_electrons = {str(element): element.common_oxidation_states for element in elements}
        electronegativity = {str(element): element.X for element in elements}
        atomic_radius = {str(element): element.atomic_radius for element in elements}
        ionization_energy = {str(element): element.ionization_energies for element in elements}
        electronic_configuration = {str(element): element.full_electronic_structure for element in elements}
        
        #Pagrindinė informacija apie kristalą
        lattice = structure.lattice
        formula = structure.composition.reduced_formula
        space_group_symbol, space_group_number = \
            structure.get_space_group_info() or (None, None)

        #Papildoma informacija apie kristalą
        volume = structure.volume
        charge = structure.charge
        weight = structure.composition.weight
        formula_weighted = structure.composition.to_weight_dict
        atom_counts = dict(structure.composition.as_dict())
        distances = structure.distance_matrix
        species = [str(site.species) for site in structure.sites]
        
        core_data = {
            'codid': cif_file_name.split('.')[0],
            'sg_symbol': space_group_symbol,
            'sg_number': space_group_number,
            'reduced_formula': formula,
            'lattice_angle_alpha': lattice.angles[0],
            'lattice_angle_beta': lattice.angles[1],
            'lattice_angle_gamma': lattice.angles[2],
            'lattice_param_a': lattice.a,
            'lattice_param_b': lattice.b,
            'lattice_param_c': lattice.c,
            'formula_weighted': formula_weighted,
            'atom_counts': atom_counts,
            'volume': volume,
            'charge': charge,
            'weight': weight,
            'valence_electrons': valence_electrons,
            'electronegativity': electronegativity,
            'atomic_radius': atomic_radius,
            'ionization_energy': ionization_energy,
            'electronic_configuration': electronic_configuration,
        }
        
        auxiliary_data = {
            'codid': cif_file_name.split('.')[0],
            'distance_matrix': distances,
            'species' : species
        }
        
        return core_data, auxiliary_data
    except Exception as e:
        print(f"Error processing {cif_file_name}: {e}")
        return None, None

def process_files(
    cif_files_content, output_csv_hdf5_species_name
    ):
    with(
        open(
            output_csv_hdf5_species_name + '_core.csv', 'w', newline=''
            ) as csvfile,
        h5py.File(
            output_csv_hdf5_species_name + '_aux.hdf5', 'a'
            ) as hdf5file, 
        h5py.File(
            output_csv_hdf5_species_name + '_species.hdf5', 'a'
            ) as hdf5file_species
        ):
        
        core_fieldnames = [
            'codid', 'sg_symbol', 'sg_number', 'lattice_angle_alpha', 
            'lattice_angle_beta', 'lattice_angle_gamma', 'lattice_param_a',
            'lattice_param_b', 'lattice_param_c', 'reduced_formula', 
            'formula_weighted', 'atom_counts', 'volume', 'charge', 'weight',
            'valence_electrons', 'electronegativity', 'atomic_radius',
            'ionization_energy', 'electronic_configuration'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=core_fieldnames)
        writer.writeheader()
        
        with ThreadPoolExecutor() as executor:
        #Naudojam multiprocessing, kad paspartinti duomenų apdorojimą
        
            tasks = [(name, content) for name, content in cif_files_content.items()]
            for core_data, aux_data in tqdm(
                executor.map(lambda p: read_file(*p), tasks), 
                total=len(tasks), desc="Processing CIF files"
                ):
                if core_data and aux_data:                    
                    writer.writerow(core_data)
                    hdf5_group = hdf5file.create_group(aux_data['codid'])
                    hdf5_group.create_dataset(
                        'distance_matrix', 
                        data=aux_data['distance_matrix']
                        )
                    hdf5_group_species = hdf5file_species.create_group(
                        aux_data['codid']
                        )
                    hdf5_group_species.create_dataset(
                        'species', 
                        data=np.array(aux_data['species'], dtype='S')
                        )

if __name__ == "__main__":
    data_folder = sys.argv[1]
    output_csv_hdf5_species_name = sys.argv[2]
    if not os.path.exists(data_folder):
        raise FileNotFoundError("No data directory found")
    
    cif_files_content = cache_cif_files(data_folder)
    process_files(
        cif_files_content, output_csv_hdf5_species_name
        )
