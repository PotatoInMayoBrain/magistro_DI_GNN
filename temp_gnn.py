import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re

def one_hot_encode(atomic_numbers):
    encoder = OneHotEncoder()
    atomic_numbers = np.array(atomic_numbers).reshape(-1, 1)
    one_hot_encoded = encoder.fit_transform(atomic_numbers).toarray()
    return one_hot_encoded

def parse_position(position_str):
    match = re.match(r"([0-9.-]+)\([0-9]+\)", position_str)
    if match:
        return float(match.group(1))
    else:
        return float(position_str)

# Function to parse atomic positions list
def parse_positions_list(positions_list_str):
    positions_list = eval(positions_list_str)
    return [[parse_position(pos) for pos in pos_list] for pos_list in positions_list]


df = pd.read_csv('data.csv')    

# Replace '?' values with NaN
df = df.replace('?', np.nan)

# Drop rows with NaN values
df = df.dropna()

df['Atomic Positions'] = df['Atomic Positions'].apply(parse_positions_list)
df['Atomic Numbers'] = df['Atomic Numbers'].apply(eval)

print(df['Atomic Numbers'])


#print(df.head())

from scipy.spatial import KDTree

graphs = []
for _, row in df.iterrows():
    atomic_positions = np.array(row['Atomic Positions'])
    atomic_numbers = [num for sublist in one_hot_encode(row['Atomic Numbers']) for num in sublist]

    # Create a graph for this crystal structure
    G = nx.Graph()

    # Add nodes
    for i, atomic_number in enumerate(atomic_numbers):
        G.add_node(i, features=atomic_number)

    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(atomic_positions)

    # Find pairs of nodes within a certain distance
    pairs = tree.query_pairs(r=10)  # adjust the distance as needed

    # Add edges between the pairs of nodes
    for i, j in pairs:
        # Compute the relative position
        relative_position = atomic_positions[i] - atomic_positions[j]
        # Add the edge with the relative position as the feature
        G.add_edge(i, j, features=relative_position)

    graphs.append(G)

import matplotlib.pyplot as plt

# Select the first graph to visualize
G = graphs[0]

# Create a figure and draw the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # Compute the positions of the nodes using the spring layout algorithm
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='k')

# Draw the edge labels
edge_labels = nx.get_edge_attributes(G, 'features')
for key in edge_labels.keys():
    edge_labels[key] = str(edge_labels[key])
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

#plt.show()

