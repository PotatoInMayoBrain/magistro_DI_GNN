import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch.nn import Linear
import numpy as np



class LowerLevelGNN(torch.nn.Module):
    def __init__(self):
        super(LowerLevelGNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.classifier = Linear(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_max_pool(x, batch)  # Pooling
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
class HigherLevelGNN(torch.nn.Module):
    def __init__(self):
        super(HigherLevelGNN, self).__init__()
        self.conv1 = GCNConv(lower_level_gnn.output_dim, 64)
        self.conv2 = GCNConv(64, 64)
        self.classifier = Linear(64, num_space_groups)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_max_pool(x, batch)  # Pooling
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def preprocess_data(data):
    #Create a Data object for each crystal structure that represents the structure as a graph
    #Add node features for each atom

if __name__ == "__main__":
    lower_level_gnn = LowerLevelGNN()
    higher_level_gnn = HigherLevelGNN()
    print(lower_level_gnn)
    print(higher_level_gnn)
    print("Done")

    