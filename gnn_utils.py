'''
        Šiame faile talpinamos visos pagalbinės funkcijos, 
        kurios naudojamos GNN modelio treniravime ir testavime.
'''

import sys
import ast
import re
import h5py
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torch_geometric.utils import to_networkx, dense_to_sparse
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import optuna
from gnn_model import GNNModel
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, \
    SAGEConv, MFConv

class GNNUtils():
    def __init__(self):
        self.bravais_lattice_mapping = {
        range(1, 3): 'Triclinic',
        range(3, 16): 'Monoclinic',
        range(16, 75): 'Orthorhombic',
        range(75, 143): 'Tetragonal',
        range(143, 168): 'Trigonal',
        range(168, 195): 'Hexagonal',
        range(195, 231): 'Cubic'
        }

        self.nested_columns = [
        'formula_weighted', 'atom_counts', 'valence_electrons', 
        'electronegativity', 'atomic_radius', 'ionization_energy', 
        'electronic_configuration'
        ]

        self.non_real_elements = ['D', 'M', 'X']

        self.atomic_number_mapping = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,'O': 8,'F': 9, 
        'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 
        'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 
        'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': .5, 'Kr': 36, 'Rb': 37, 
        'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 
        'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 
        'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
        'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 
        'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 
        'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 
        'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 
        'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 
        'Md': 101, 'No': 102, 'Lr': 103,'Rf': 104,'Db': 105,'Sg': 106,'Bh': 107,
        'Hs': 108, 'Mt': 109, 'Ds': 110,'Rg': 111,'Cn': 112,'Nh': 113,'Fl': 114,
        'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118,
        }

    def find_max_value_in_electronegativity_atomic_radius(self, df):
        all_atomic_radius = []
        all_electronegativity = []
        count = 0
        for i, row_data in tqdm(
            df.iterrows(), total=len(df), desc='Processing rows'
            ):
            codid = row_data['codid']
            codid_str = str(codid)
            for col in ['atomic_radius', 'electronegativity']:
                row_data[col] = self.safe_literal_eval(row_data[col])
                try:
                    for key in row_data[col]:
                        if col == 'atomic_radius':
                            if row_data[col][key] is not None:
                                all_atomic_radius.append(row_data[col][key])
                        if col == 'electronegativity':
                            if row_data[col][key] is not None:
                                all_electronegativity.append(row_data[col][key])
                except TypeError:
                    print(
                        f'No electronegativity/atomic radius data for \
                        {codid_str}, dropping this row'
                        )
                    count += 1
                    df.drop(i, inplace=True)
                    continue
        print(f'Dropped {count} rows')
        return df, max(all_atomic_radius), max(all_electronegativity)

    def normalize_dict(self, d, min_val, max_val, new_min=0, new_max=1):
        return {
            k: (v - min_val) / (max_val - min_val) * (new_max - new_min) + 
            new_min for k, v in d.items() if v is not None
            }

    def normalize_dict_values(self, d, global_min, global_max):
        values = np.array(list(d.values()))
        normalized_values = (values - global_min) / (global_max - global_min)

        normalized_dict = {
            key: normalized_values[i] for i, key in enumerate(d.keys())
            }
        return normalized_dict

    def normalize_and_invert_list_values_in_dict(
        self, d, global_min, global_max
        ):
        normalized_dict = {}
        for key, value in d.items():
            value_array = np.array(value)
            inverted_value = 1 / (value_array + 1e-9)
            normalized_value = (
                (inverted_value - global_min) / (global_max - global_min)
                )
            normalized_dict[key] = normalized_value
        return normalized_dict

    def normalize_atomic_radius(
        self, atomic_radius, charge, global_means, 
        global_stds, global_min, global_max
        ):
        normalized_radius = {}
        for atom_type, radius in atomic_radius.items():
            key = (atom_type, charge)
            if key in global_means and global_stds[key] != 0:
                normalized_value = (
                    (radius - global_means[key]) / global_stds[key]
                    )
                normalized_radius[atom_type] = (
                    (normalized_value - global_min) / (global_max - global_min)
                    )
            else:
                normalized_radius[atom_type] = 0  # Handle cases where std is 0
        return normalized_radius

    def normalize_values(self, scaler, data, key):
        key = str(key)
        if key in data:
            data['normalized_' + key] = scaler.fit_transform(data[[key]])
        return data

    def normalize_charge(self, data, key):
        key = str(key)
        if key == 'charge':
            charge_min = data[key].min()
            charge_max = data[key].max()

            if abs(charge_min) > charge_max:
                charge_max = abs(charge_min)
            elif abs(charge_min) < charge_max:
                charge_min = -charge_max

            # Normalize negative charges to [-1, 0]
            negative_mask = data[key] < 0
            data.loc[negative_mask, 'normalized_' + key] = (
                data.loc[negative_mask, key] / abs(charge_min)
            )
            # Normalize positive charges to [0, 1]
            positive_mask = data[key] > 0
            data.loc[positive_mask, 'normalized_' + key] = (
                data.loc[positive_mask, key] / charge_max
            )
            # Keep 0 as 0
            zero_mask = data[key] == 0
            data.loc[zero_mask, 'normalized_' + key] = 0

            return data
        else:
            raise ValueError(f"Invalid key, expected 'charge' but got '{key}'")

    def normalize_species(self, data):
        return (data - 1) / (118 - 1)

    def parse_species(self, species_list):
        parsed_species = []
        for species in species_list:
            species_dict = {}
            for item in species.decode().split():
                element = re.match(r'^[A-Za-z]+', item).group(0)
                prob = (
                    float(re.findall(r'\d*\.?\d+', item)[0]) if re.findall(
                        r'\d*\.?\d+', item) else 1.0
                    )
                species_dict[element] = prob
            parsed_species.append(species_dict)
        return parsed_species

    def extract_unique_elements(self, hdf5_species_path):
        unique_elements = set()
        element_pattern = re.compile(r'^[A-Za-z]+')

        # Extract unique elements from species.hdf5
        with h5py.File(hdf5_species_path, 'r') as hdf5_species:
            for codid in tqdm(
                hdf5_species.keys(), total=len(hdf5_species.keys()), 
                desc='Extracting unique elements'
                ):
                codid_str = str(codid)
                species_group = hdf5_species[codid_str]
                species_list = species_group['species']
                for species in species_list:
                    for item in species.decode().split():
                        match = element_pattern.match(item)
                        if match:
                            unique_elements.add(match.group(0))
        return sorted(list(unique_elements))

    def encode_species_atomic_number(self, parsed_species):
        encoded_species = []
        for species in parsed_species:
            encoded = 0
            for element, _ in species.items():
                atomic_number = self.atomic_number_mapping.get(element, 0)
                encoded += atomic_number
                break
            encoded_species.append(encoded)
        return np.array(encoded_species)

    def pad_list(self, lst, length, padding_value=0):
        return lst + [padding_value] * (length - len(lst))

    def has_none_value(self, atomic_radius_dict):
        return any(radius is None for radius in atomic_radius_dict.values())

    def extract_charge(self, key):
        match = re.search(r'(\d+)([+-])', key)
        if match:
            number = int(match.group(1))
            sign = match.group(2)
            return number if sign == '+' else -number
        return 0

    def normalize_charges(self, d):
        charges = {k: self.extract_charge(k) for k in d.keys()}
        max_abs_charge = 10
        normalized_charges = {k: v / max_abs_charge for k, v in charges.items()}
        return normalized_charges

    def clean_atom_key(self, atom_key):
        return re.sub(r'[^a-zA-Z]', '', atom_key)

    def clean_dict_keys(self, input_dict):
        return {
            self.clean_atom_key(key): value for key, value in input_dict.items()
            }

    def has_none_values(self, d):
        return any(v is None for v in d.values())

    def safe_literal_eval(self, val):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None

    def one_hot_encode_bravais(self, dataframe):
        one_hot = pd.get_dummies(dataframe['sg_bravais'])
        dataframe = dataframe.drop('sg_bravais',axis = 1)
        dataframe = dataframe.join(one_hot)
        return dataframe

    def draw_graph(self, data):
        G = to_networkx(data, to_undirected=True)
        plt.figure(figsize=(8, 8))
        nx.draw(
            G, with_labels=True, node_size=500, node_color='skyblue', 
            font_size=10, font_color='black', font_weight='bold'
            )
        plt.show()

    def plot_losses(self, train_losses, val_losses, save=False, name='Train_val_loss.eps'):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b', label='Training loss')
        plt.plot(epochs, val_losses, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if save == True:
          plt.savefig(name)
        else:
          plt.show()
        
    def map_to_bravais(self, sg_number):
        for key in self.bravais_lattice_mapping:
            if sg_number in key:
                return self.bravais_lattice_mapping[key]
        return 'Unknown'
    
    
    def drop_non_real_elements(self, row_data, species, non_real_elements):
        codid_str = str(row_data['codid'])
        if any(
            non_real_element in species 
            for non_real_element in non_real_elements
            ):
            print(f'Non-real element found in {codid_str}, dropping this row')
            return True
        return False
    
    
    def create_graph_object(
        self, row_data, index, hdf5_data, 
        hdf5_species, non_real_el=None
        ):
        if non_real_el is None:
            non_real_el = self.non_real_elements
        
        codid_str = str(row_data['codid'])
        group = hdf5_data[codid_str]
        group_species = hdf5_species[codid_str]
        
        adjacency_matrix = group['distance_matrix'][:]
        species = group_species['species'][:]
        parsed_species = self.parse_species(species)
        
        if self.drop_non_real_elements(row_data, parsed_species, non_real_el):
            print(f'Row {index} dropped')
            return None
        
        formula_weighted = (self.clean_dict_keys(
            self.safe_literal_eval(row_data['formula_weighted'])
            ))
        valence_electrons = self.clean_dict_keys(
            row_data['valence_electrons_normalized']
            )
        atomic_radius = self.clean_dict_keys(
            row_data['atomic_radius_normalized']
            )
        electronegativity = self.clean_dict_keys(
            row_data['electronegativity_normalized']
            )
        
        node_features = []
        for species in parsed_species:
            for atom, _ in species.items():
                radius = atomic_radius[atom]
                electronegativity_temp = electronegativity[atom]
                valence = valence_electrons[atom]
                normalized_species = self.normalize_species(
                    self.encode_species_atomic_number([species])
                    ).item()
                formula_weighted_temp = formula_weighted[atom]
                features = [
                    normalized_species, radius, electronegativity_temp, valence, 
                    formula_weighted_temp
                    ]
                node_features.append(features)
        
        lattice_params = np.array(
            [
                row_data['lattice_param_a'], 
                row_data['lattice_param_b'], 
                row_data['lattice_param_c']
                ]
            )
        lattice_angles = np.array(
            [
                row_data['lattice_angle_alpha'], 
                row_data['lattice_angle_beta'], 
                row_data['lattice_angle_gamma']
                ]
            )
        scaler = MinMaxScaler()
        lattice_params_normalized = scaler.fit_transform(
            lattice_params.reshape(-1, 1)
            ).flatten()
        lattice_angles_normalized = scaler.fit_transform(
            lattice_angles.reshape(-1, 1)
            ).flatten()
        
        label = row_data[
            [
                'Cubic', 'Hexagonal', 'Monoclinic', 'Orthorhombic', 
                'Tetragonal', 'Triclinic', 'Trigonal'
                ]
            ].astype(float).values
        
        edge_index, edge_attr = dense_to_sparse(
            torch.tensor(adjacency_matrix, dtype=torch.float)
            )
        
        charge = row_data['normalized_charge']
        volume = row_data['normalized_volume']
        weight = row_data['normalized_weight']
        lattice_params_normalized_flat = lattice_params_normalized.flatten()
        lattice_angles_normalized_flat = lattice_angles_normalized.flatten()
        
        graph_features = np.concatenate(
            [
                lattice_params_normalized_flat,
                lattice_angles_normalized_flat,
                np.array([volume]),
                np.array([charge]),
                np.array([weight])
                ]
            )
        
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(label, dtype=torch.float),
            graph_features = torch.tensor(graph_features, dtype=torch.float),
            codid = codid_str
            )
        
        return data
    
    def find_non_real_elements(self, unique_elements):
        
        known_elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 
        'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 
        'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
        'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 
        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 
        'Lv', 'Ts', 'Og'
        ]
        real = [el for el in unique_elements if el in known_elements]
        non_real = [el for el in unique_elements if el not in known_elements]
        print("Real elements:", real)
        print("Non-real elements:", non_real)
        print(\
            "Real elements not in species: ",\
            print(set(real) - set(known_elements))
            )
    
    def misc_preprocess(
        self, 
        df: pd.DataFrame, 
        hdf5_file_species: str, 
        global_min_radius: float,
        global_max_radius: float,
        global_min_electronegativity: float, 
        global_max_electronegativity: float,
        do_unique_elements: bool = False
        ):
        df['atomic_radius_dict'] = df['atomic_radius'].apply(
            self.safe_literal_eval
            )
        df['electronegativity_dict'] = df['electronegativity'].apply(
            self.safe_literal_eval
            )
        df['valence_electrons_dict'] = df['valence_electrons'].apply(
            self.safe_literal_eval
            )

        df = df[df['atomic_radius_dict'].notnull()]
        df = df[df['electronegativity_dict'].notnull()]
        df = df[df['valence_electrons_dict'].notnull()]

        df = df[~df['atomic_radius_dict'].apply(self.has_none_values)]
        df = df[~df['electronegativity_dict'].apply(self.has_none_values)]
        df = df[~df['valence_electrons_dict'].apply(self.has_none_values)]

        df['atomic_radius_normalized'] = df['atomic_radius_dict'].apply(
            lambda d: self.normalize_dict(
                d, global_min_radius, global_max_radius
                )
            )
        df['electronegativity_normalized'] = df['electronegativity_dict'].apply(
            lambda d: self.normalize_dict(
                d, global_min_electronegativity, global_max_electronegativity
                )
            )
        df['valence_electrons_normalized'] = df['valence_electrons_dict'].apply(
            lambda d: self.normalize_charges(d)
            )

        df = self.normalize_values(MinMaxScaler(), df, 'volume')
        df = self.normalize_values(MinMaxScaler(), df, 'weight')
        df = self.normalize_charge(df, 'charge')
        if do_unique_elements:
            unique_elements = self.extract_unique_elements(hdf5_file_species)
            non_real_el = self.find_non_real_elements(
                unique_elements=unique_elements
                )
            return df, non_real_el, unique_elements
        else:
            return df
        
    def plot_fold_losses(fold_results):
        # Extract epoch-wise results
        epochs = [result['epoch'] for result in fold_results]
        train_losses = [result['train_loss'] for result in fold_results]
        val_losses = [result['val_loss'] for result in fold_results]
        folds = [result['fold'] for result in fold_results]

        # Initialize the plot
        plt.figure(figsize=(12, 6))
        for fold in sorted(set(folds)):
            fold_epochs = [
                epoch for i, epoch in enumerate(epochs) if folds[i] == fold
                ]
            fold_train_losses = [
                train_losses[i] for i, epoch in enumerate(epochs) 
                if folds[i] == fold
                ]
            fold_val_losses = [
                val_losses[i] for i, epoch in enumerate(epochs) 
                if folds[i] == fold
                ]

            plt.plot(
                fold_epochs, fold_train_losses, label=f'Fold {fold} Train Loss'
                )
            plt.plot(
                fold_epochs, fold_val_losses, label=f'Fold {fold} Val Loss', 
                linestyle='--'
                )

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses per Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def load_model(model_class, model_path, device):
        model = model_class(num_classes=7, dropout_rate=0).to(device)
        model.load_state_dict(torch.load(model_path))
        return model


class GNNTrain_funcs():
    def __init__(self, k_folds=5):
        self.k_folds = k_folds
        
    def train(
        self, dataset, model_class, device, optimizer_class, 
        criterion, num_workers, model_num_classes, dropout, num_layers=3,
        hidden_channels=[64,128,256], batch_size=32, epochs = 10,**optim_kwargs, 
        ):
        kf = KFold(n_splits=self.k_folds, shuffle=True)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(
            kf.split(range(len(dataset)))
            ):
            print(f"Fold {fold + 1}/{self.k_folds}")

            
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]
            
            
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers, pin_memory=True, prefetch_factor=2,
                persistent_workers=True
                )
            val_loader = DataLoader(
                val_data, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, prefetch_factor=2,
                persistent_workers=True
                )
            
            model = model_class(
                model_num_classes, dropout, num_layers, hidden_channels
                ).to(device)
            optimizer = optimizer_class(model.parameters(), **optim_kwargs)
            
            for epoch in tqdm(range(epochs)):
                print(f"Epoch {epoch + 1}/{epochs}")
                avg_train_loss, train_acc, train_prec, train_rec, train_f1 = \
                    self.train_one_fold(
                    train_loader, model, device, optimizer, criterion
                )
                
                avg_val_loss, val_acc, val_prec, val_rec, val_f1 = \
                    self.validate_one_fold(
                    val_loader, model, device, criterion
                )
            
                fold_results.append({
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_acc': train_acc,
                    'train_prec': train_prec,
                    'train_rec': train_rec,
                    'train_f1': train_f1,
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc,
                    'val_prec': val_prec,
                    'val_rec': val_rec,
                    'val_f1': val_f1
                })

        return fold_results

    def train_one_fold(self,train_loader, model, device, optimizer, criterion):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for data in tqdm(train_loader, total=len(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            classification_output = model(data)    
            loss = criterion(classification_output, data.y) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = classification_output.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_loss = running_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, precision, recall, f1

    def validate_one_fold(self, val_loader, model, device, criterion):
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in tqdm(val_loader, total=len(val_loader)):
                data = data.to(device)
                classification_output = model(data)
                loss = criterion(classification_output, data.y)
                running_loss += loss.item()

                preds = classification_output.argmax(dim=1).cpu().numpy()
                labels = data.y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_loss = running_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, precision, recall, f1

    def test(self, test_loader, model, device, criterion, top_n=3):
        model.eval()
        running_loss = 0.0
        top_n_correct = 0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                classification_output = model(data)
                loss = criterion(classification_output, data.y)
                running_loss += loss.item()

                preds = classification_output.argmax(dim=1).cpu().numpy()
                labels = data.y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

                # Top-n accuracy calculation
                top_n_preds = torch.topk(classification_output, top_n, dim=1).indices.cpu().numpy()
                for i in range(len(labels)):
                    if labels[i] in top_n_preds[i]:
                        top_n_correct += 1
                total_samples += len(labels)

        avg_loss = running_loss / len(test_loader)
        top_n_accuracy = top_n_correct / total_samples

        return avg_loss, all_preds, all_labels, top_n_accuracy
    
    def objective(self, trial, train_loader, val_loader):
    # Define the hyperparameters to tune
        hidden_channels = [
            trial.suggest_int("hidden_channels_1", 64, 1024),
            trial.suggest_int("hidden_channels_2", 128, 2048),
            trial.suggest_int("hidden_channels_3", 256, 4096),
            #trial.suggest_int("hidden_channels_3", 512, 2048),
            #trial.suggest_int("hidden_channels_3", 1024, 4096),
        ]
        fc_out_channels = [
            trial.suggest_int("fc_out_channels_1", 256, 4096),
            trial.suggest_int("fc_out_channels_2", 128, 2048),
            trial.suggest_int("fc_out_channels_3", 64, 1024)
        ]
        conv_layer = [SAGEConv, SAGEConv, MFConv]  # Fixed for simplicity
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.8)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the model
        model = GNNModel(
            num_classes=7, hidden_channels=hidden_channels, 
            conv_layer=conv_layer, dropout_rate=dropout_rate,
            fc_out_channels=fc_out_channels
            ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        criterion = torch.nn.CrossEntropyLoss()

        # Train and validate the model
        for epoc in range(7):
            self.train_one_fold(train_loader, model, device, optimizer, criterion)
        val_loss, _, _, _, _ = self.validate_one_fold(
            val_loader, model, device, criterion
            )
        
        return val_loss

if __name__ == '__main__':
    print('This script is not meant to be run directly')
    pass