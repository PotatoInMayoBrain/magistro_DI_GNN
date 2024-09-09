import sys
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Step 1: Data Loading
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Extract relevant data from the row
        codid = row['cod_id']
        sg = row['sg']
        lattice_a = row['lattice_a']
        lattice_b = row['lattice_b']
        lattice_c = row['lattice_c']
        lattice_alpha = row['lattice_alpha']
        lattice_beta = row['lattice_beta']
        lattice_gamma = row['lattice_gamma']
        distances = [float(d) for d in row['adjacency_matrix'].split()]  # Convert distances to list of floats
        #print("\ndistances: ", distances, "\ncodid: ",codid ,"\nsg: ", sg)

        adj_matrix = create_adjacency_matrix(distances)
        adj_matrix = sorted(adj_matrix, key=len, reverse=True)
        adj_matrix = append_negative_to_adjacency_matrix(adj_matrix)

        x, edge_index, edge_attr = preprocess_data(adj_matrix, codid)  # Preprocess distances into edge_index and edge_attr
        y = torch.tensor([sg], dtype=torch.long)  # Target label (space group)
        #print(f"X SIZE: {x.size()}, X VALUES: \n")
        #print("Edge index: ", edge_index, "\nEdge attributes: ", edge_attr, "\nX: ", x, "\nY: ", y)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def create_adjacency_matrix(distances):
    adjacency_matrix = []
    current_row = []

    for distance in distances:
        if distance == -1:
            if current_row:
                adjacency_matrix.append(current_row)
                current_row = []
        else:
            current_row.append(distance)

    if current_row:
        adjacency_matrix.append(current_row)

    return adjacency_matrix


def append_negative_to_adjacency_matrix(distances):
    max_length = max(len(row) for row in distances)  # Find the maximum length of lists

    adjacency_matrix = []
    for row in distances:
        while len(row) < max_length:  # Add -1 to shorter lists
            row.append(-1.0)
        adjacency_matrix.append(row)

    return adjacency_matrix

def preprocess_data(adj_matrix, codid):
    edge_list = []
    edge_attr = []
    num_atoms = len(adj_matrix)

    # Create a tensor for node features
    x = torch.ones((num_atoms, 1))  # Assuming one feature per atom

    for src_atom_idx in range(num_atoms):
        for dest_atom_idx, distance in enumerate(adj_matrix[src_atom_idx]):
            if distance != -1 and src_atom_idx != dest_atom_idx:  # Ensure it's not a self-loop
                edge_list.append((src_atom_idx, dest_atom_idx))
                edge_attr.append(distance)

    # Convert to a tensor of shape [2, num_edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

    #print("X size for one is: ", x.size())
    return x, edge_index, edge_attr



# Step 2: Define GNN Model
class GNNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNNModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.conv1 = GCNConv(num_features, 64)  # Input feature size to 64 output features
        self.conv2 = GCNConv(64, 128)            # 64 input feature, 128 output features
        self.conv3 = GCNConv(128, 256)           # 128 input feature, 256 output features
        self.lin1 = Linear(256, 128)             # Linear layer for intermediate processing
        self.lin2 = Linear(128, num_classes)     # Linear layer for final output
        
        
    def forward(self, x, edge_index, edge_attr):
        #print(f"D")
        #print(f"DATA.X YRA: {data.x},")
        #print(f"DATA.EDGE_INDEX YRA: {data.edge_index},")
        #print(f"DATA.EDGE_ATTR YRA: {data.edge_attr},")
        #print(f"DATA.Y YRA: {data.y},")
        #print(f"DATA.BATCH YRA: {data.batch}")
        #print(f"DATA.PTR YRA: {data.ptr}")
        
        #print("\nEdge index size: ", edge_index.size(), "\nEdge attribute size: ", edge_attr.size(), "\nX size: ", x.size())
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = global_max_pool(x, batch=None)  # Global max pooling
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizer, num_epochs):
    model.train()
    loss_return = []
    for epoch in range(num_epochs):
        total_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            data.to(model.device)
            optimizer.zero_grad()
            #print("DATA.X: ", data.x)
            out = model(data.x, data.edge_index, data.edge_attr)
            #print ("OUT SIZE: ", out.size(), "DATA.Y SIZE: ", data.y.size())
            loss = F.nll_loss(out, data.y)  # Negative log-likelihood loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
        loss_return.append(total_loss / len(train_loader))
    return loss_return

def test(model, test_loader):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for data in test_loader:
            data.to(model.device)
            out = model(data.x, data.edge_index, data.edge_attr)
            batch_size = data.y.size(0)  # Get the batch size of the target tensor
            loss = F.nll_loss(out, data.y, reduction='sum')  # Negative log-likelihood loss
            total_loss += loss.item()
            total_samples += batch_size
    return total_loss / total_samples

#create a draw_graph function to take the trained model and draw the graph
def draw_graph(model):
    # Convert the model to a networkx graph
    graph = to_networkx(model)
    
    # Draw the graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    plt.show()

# Step 4: Main Execution
if __name__ == "__main__":
    
    csvfile = sys.argv[1]
    # Load dataset
    dataset = CustomDataset(csvfile)

    num_features = 1
    num_classes = 230  # Number of space groups
    epochs = 15

    # Split dataset into train and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2)

    # Define data loaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, shuffle=False)
    
    #print(f"TRAIN LOADER PRINTOUTAS: {train_loader.x}")
    # Define model and optimizer
    
    model = GNNModel(num_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #with tqdm(range(epochs), desc='Training Loop') as pbar:
        # Training loop
        
    #for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, epochs)
    test_loss = test(model, test_loader)
    # Plot the MSE loss against the epochs
    print("TEST LOSS: ", test_loss)
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_loss, label='train Loss')
    #plt.plot(range(epochs), test_loss, label='test Loss')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    #draw_graph(model)
