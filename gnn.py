import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import csv

# Define the Graph Neural Network model
class CrystalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CrystalGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load data from the adjacency matrix CSV file
def load_data_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        adjacency_matrix = list(csv.reader(csvfile, delimiter=','))
    adjacency_matrix = np.array(adjacency_matrix, dtype=np.float64)
    num_nodes = adjacency_matrix.shape[0]
    edge_index = np.array(np.nonzero(adjacency_matrix)).T
    x = torch.tensor(np.eye(num_nodes), dtype=torch.float)  # Identity matrix as node features
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

if __name__ == "__main__":
    # Read input CSV file path from command-line arguments
    input_csv_file = sys.argv[1]

    # Load data from the adjacency matrix CSV file
    data = load_data_from_csv(input_csv_file)

    # Define model dimensions
    input_dim = data.num_features
    hidden_dim = 64
    output_dim = 2  # You can adjust this based on your problem

    # Create the CrystalGNN model
    model = CrystalGNN(input_dim, hidden_dim, output_dim)

    # Example forward pass
    output = model(data)

    # Write output to stdout
    print(output)

