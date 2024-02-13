import sys
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd 

def create_graph_from_adjacency_matrix(adj_matrix):
    G = nx.Graph()
    num_nodes = len(adj_matrix)

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # Add edges to the graph based on the adjacency matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = adj_matrix[i][j]
            if distance != -1:  # Only add edges for non-None distances
                G.add_edge(i, j, weight=float(distance))

    return G

# Read the adjacency matrix from the file
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

# Example usage
filename = sys.argv[1]  # Specify the path to your file

data = pd.read_csv(filename)
print(data)
#for idx in range(len(data)):
idx = 43
row = data.iloc[idx]
distances = [float(d) for d in row['distances'].split()]

adj_matrix = create_adjacency_matrix(distances)
adj_matrix = sorted(adj_matrix, key=len, reverse=True)
adj_matrix = append_negative_to_adjacency_matrix(adj_matrix)
graph = create_graph_from_adjacency_matrix(adj_matrix)

# You can now use NetworkX's functions to analyze or visualize the graph
print("Nodes of the graph:", graph.nodes())
#print("Edges of the graph:", graph.edges(data=True))


# Draw the graph
pos = nx.spring_layout(graph)  # positions for all nodes
nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", width=2)

# Draw edge labels
edge_labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

# Show the graph
plt.title("Graph with Unique Edges")
plt.show()