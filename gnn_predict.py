import os
import requests
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gnn_utils import GNNUtils  # Assuming utils contains the load_model function
import h5py
import pandas as pd
from pymatgen_cif_parse import read_file, process_files, cache_cif_files
from gnn_model import GNNModel, GraphDataset

class helperfunc:
    def __init__(self):
        self.utils = GNNUtils()
    
    def download_cif(self, cod_id, save_path):
        url = f"http://www.crystallography.net/cod/{cod_id}.cif"
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)

    def classify_graph(self, model, graph, device):
        model.eval()
        graph = graph.to(device)
        with torch.no_grad():
            output = model(graph)
            true_label = graph.y
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_label = torch.max(probabilities, dim=1)
        return predicted_label.item(), confidence.item(), true_label

    def create_graph(self, csv_path, hdf5_path, species_hdf5_path):
        csv_df = pd.read_csv(csv_path)
        hdf5_data = h5py.File(hdf5_path, 'r')
        species_hdf5_data = h5py.File(species_hdf5_path, 'r')
        csv_df['sg_bravais'] = csv_df['sg_number'].apply(
            (lambda x: self.utils.map_to_bravais(sg_number=x)
            )
        )
        csv_df = self.utils.one_hot_encode_bravais(csv_df)
        global_min_radius = 0
        global_max_radius = 2.6 # Francium
        global_min_electronegativity = 0
        global_max_electronegativity = 3.98 # Fluorine
        
        csv_df, non_real, _ = self.utils.misc_preprocess(
            csv_df, species_hdf5_path, global_min_radius, global_max_radius, 
            global_min_electronegativity, global_max_electronegativity,
            do_unique_elements=True
            )
        graphs = []
        for i, row in csv_df.iterrows():
            data = self.utils.create_graph_object(
                row, i, hdf5_data, species_hdf5_data, non_real
                )
            graphs.append(data)
        return graphs

def main(cod_id):
    utils = GNNUtils()
    helper = helperfunc()
    str_cod_id = str(cod_id)
    cif_path = f"{cod_id}.cif"
    csv_path = f"{cod_id}.csv"
    hdf5_path = f"{cod_id}_aux.hdf5"
    species_hdf5_path = f"{cod_id}_species.hdf5"
    
    # Step 1: Download the .cif file
    helper.download_cif(cod_id, cif_path)
    dir_path = os.getcwd()
    
    # Step 2: Create CSV and HDF5 files
    with open(os.path.join(dir_path, cif_path), 'r') as f:
        process_files(f.read(), cod_id)
    
    # Step 3: Create a graph object
    
    graph = helper.create_graph(csv_path, hdf5_path, species_hdf5_path)
    
    # Step 4: Load the pretrained model
    model_class = GNNModel
    model_path = 'Sixth_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = utils.load_model(
        model_class, model_path, device, model_name='arch-1'
        )
    
    # Step 5: Classify the graph and show confidence
    predicted_label, confidence, true = helper.classify_graph(
        model, graph, device
        )
    print(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}\
        True Label: {true}")

if __name__ == "__main__":
    user_input = input(
        "Enter codid that exists in COD: "
        )
    
    try:
        main(user_input)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)