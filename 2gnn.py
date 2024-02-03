import sys
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn import Linear
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm


# Step 1: Data Loading
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Extract relevant data from the row
        codid = row['codid']
        sg = row['sg']
        distances = [float(d) for d in row['distances'].split()]  # Convert distances to list of floats
        #print("\ndistances: ", distances, "\ncodid: ",codid ,"\nsg: ", sg)
        edge_index, edge_attr = preprocess_data(distances, codid)  # Preprocess distances into edge_index and edge_attr
        print(f"EDGE INDEX LENGTH (USED TO SET X SIZE): {len(edge_index)} \nEDGE INDEX SIZE: {edge_index.size()}, \nEDGE ATTR SIZE: {edge_attr.size()}")
        x = torch.ones(len(edge_index), 1)  # Node features (optional, set to 1 for simplicity)
        y = torch.tensor([sg], dtype=torch.long)  # Target label (space group)
        print(f"X SIZE: {x.size()}, X VALUES: {x}\n")
        #print("Edge index: ", edge_index, "\nEdge attributes: ", edge_attr, "\nX: ", x, "\nY: ", y)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def preprocess_data(distances, codid):
    # Initialize edge lists and edge attributes
    edge_list = []
    edge_attr = []
    
    #Find the indices of -1 in the distances list
    separator_indices = [i for i, value in enumerate(distances) if value == -1]
    #print(f"\nSeparator_indices: {separator_indices}")
    #Iterate over the pairs of atoms
    start_idx = 0
    tqdm_end = len(distances)
    with tqdm(desc = f"Constructing edges and edge attributes for codid: {codid}", total = tqdm_end) as pbar:
        for end_idx in separator_indices:
            # Extract distances for the current pair of atoms
            current_distances = distances[start_idx:end_idx]
            
        # Construct edges and edge attributes
            num_nodes = len(current_distances)
            
        # Construct edges and edge attributes
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    #print(f"I verte: {i} \nJ verte: {j}")
                    # Construct undirected edges
                    edge_list.append((i, j))
                    edge_list.append((j, i))
                    # Assign edge attributes (distances)
                    #print(f"CURRENT DISTANCES FOR I: {current_distances[i]}, \nCURRENT DISTANCES FOR J: {current_distances[j]}\n")
                    edge_attr.append(current_distances[i])
                    edge_attr.append(current_distances[j])  # Symmetric distances for undirected edges

            # Update the starting index for the next pair of atoms
            start_idx = end_idx + 1
            #print(f"EDGE LIST: {edge_list}")
            pbar.update(1)

    # Convert edge attributes to tensor
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

    # Convert edge list to edge index tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index, edge_attr



# Step 2: Define GNN Model
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(1, 64)  # 1 input feature, 64 output features
        self.conv2 = GATConv(64, 6)   # 64 input features, 6 output classes (space groups)
        self.lin = Linear(6, 6)       # Linear layer for final output

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        print("\nEdge index size: ", edge_index.size(), "\nEdge attribute size: ", edge_attr.size(), "\nX size: ", x.size())
        print("\nEdge index: ", edge_index, "\nEdge attribute: ", edge_attr, "\nX: ", x)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)  # Global mean pooling
        x = F.relu(self.lin(x))
        return F.log_softmax(x, dim=1)


# Step 3: Training Loop
def train():
    model.train()
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for data in pbar:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': loss.item()})
    pbar.close()

epochs = 1
# Step 4: Main Execution
if __name__ == "__main__":
    csvfile = sys.argv[1]
    # Load dataset
    dataset = CustomDataset(csvfile)

    # Split dataset into train and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2)

    # Define data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Define model and optimizer
    model = GNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with tqdm(desc='Training Loop', total=epochs) as pbar:
        # Training loop
        for epoch in range(epochs):
            train()
            print(f"Epoch [{epoch+1}/{epochs}], Loss:{loss.item():.4f}")
