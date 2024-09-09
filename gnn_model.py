'''Šiame faile aprašytas GNN modelis, kuris naudojamas klasifikavimui.'''

from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, \
    GraphSizeNorm, GraphNorm, SAGEConv, GINConv, MFConv, CGConv
import torch
import torch.nn.functional as F


class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        super(GraphDataset, self).__init__()
        self.graphs = graphs
        self.labels = labels
    
    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        graph = self.graphs[idx]
        graph.y = torch.tensor(
            self.labels[idx], dtype=torch.long
            ).clone().detach()
        return graph

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer, dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_layer(in_channels, hidden_channels)
        self.conv2 = conv_layer(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_norm1 = GraphNorm(hidden_channels)
        self.batch_norm2 = GraphNorm(hidden_channels)
        
        self.residual = torch.nn.Linear(
            in_channels, hidden_channels
            ) if in_channels != hidden_channels else None
        
    def forward(self, x, edge_index):
        residual = x
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        if self.residual:
            residual = self.residual(residual)
        x = x + residual
        x = F.relu(x)
        return x

class GNNModel(torch.nn.Module):
    def __init__(
        self, num_classes=7, dropout_rate=0.5, num_layers=3, 
        hidden_channels=[64, 128, 256], conv_layer=[GATConv, GATConv, GATConv],
        fc_out_channels=[4096, 2048, 1024]
        ):
        super(GNNModel, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(None)
        
        self.conv_layer = conv_layer
        self.fc_out_channels = fc_out_channels

        for i in range(1, num_layers):
            self.blocks.append(
                ResidualBlock(
                    in_channels=hidden_channels[i-1], 
                    hidden_channels=hidden_channels[i],
                    conv_layer=conv_layer[i], dropout=dropout_rate
                    )
                )
        self.fc1 = torch.nn.Linear(hidden_channels[-1] + 1, fc_out_channels[0])
        self.fc2 = torch.nn.Linear(fc_out_channels[0], fc_out_channels[1])
        self.fc3 = torch.nn.Linear(fc_out_channels[1], fc_out_channels[2])
        self.classification = torch.nn.Linear(fc_out_channels[2], num_classes)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        self.first_block_initialized = False
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if not self.first_block_initialized:
            in_channels = x.size(1)
            self.blocks[0] = ResidualBlock(
                in_channels=in_channels, hidden_channels=self.hidden_channels[0],
                conv_layer=self.conv_layer[0], dropout=self.dropout_rate
            ).to(x.device)
            self.first_block_initialized = True
        
        for block in self.blocks:
            x = block(x, edge_index)

        x = global_max_pool(x, batch)
        graph_features = data.graph_features.unsqueeze(0).expand(x.size(0), -1)
        x = torch.cat([x, graph_features], dim=1)
        
        if self.fc1 is None:
            self.fc1 = torch.nn.Linear(
                x.size(1), self.fc_out_channels[0]
                ).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        if self.fc2 is None:
            self.fc2 = torch.nn.Linear(
                x.size(1), self.fc_out_channels[1]
                ).to(x.device)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        if self.fc3 is None:
            self.fc3 = torch.nn.Linear(
                x.size(1), self.fc_out_channels[2]
                ).to(x.device)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        if self.classification is None:
            self.classification = torch.nn.Linear(
                x.size(1), self.num_classes
                ).to(x.device)
        classification_output = self.classification(x)
        
        return classification_output 