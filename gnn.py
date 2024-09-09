from gnn_utils import GNNUtils, GNNTrain_funcs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib qt
from tqdm.autonotebook import tqdm
import h5py
import ast
import re
import platform
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score
import networkx as nx

import torch
from torch_geometric.data import Dataset


from torch_geometric.utils import dense_to_sparse, to_networkx
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

from torch.optim import Adam

import warnings
warnings.filterwarnings('ignore')

from gnn_utils import GNNUtils, GNNTrain_funcs
from gnn_model import GraphDataset, GNNModel

print('Imports done')


if __name__ == '__main__':
    
    utils = GNNUtils()
    train_funcs = GNNTrain_funcs()
    
    #Load data from files
    try:
        print('Reading and loading data...')
        csv_file = sys.argv[1]
        df = pd.read_csv(csv_file)

        hdf5_file = sys.argv[2]
        hdf5_data = h5py.File(hdf5_file, 'r')

        hdf5_file_species = sys.argv[3]
        hdf5_species = h5py.File(hdf5_file_species, 'r')
    except IndexError:
        print("Usage: python gnn.py <csv_file> <hdf5_file> <hdf5_species_file>")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    
    df['sg_bravais'] = df['sg_number'].apply(
        (lambda x: utils.map_to_bravais(sg_number=x)
        )
    )
    df = utils.one_hot_encode_bravais(df)
    
    global_min_radius = 0
    global_max_radius = 2.6 # Francium
    global_min_electronegativity = 0
    global_max_electronegativity = 3.98 # Fluorine

    count = 0
    bad_codids = []
        # Laukiam vartotojo pasirinkimo
    user_input = input(
        "Do you want to search for unique elements? 1 (Yes) / 0 (No): "
        )
    try:
        do_unique_elements = bool(int(user_input))
    except:
        print("Invalid input. Using default value of no")
    
    print('Preprocessing data...')
    if do_unique_elements:
        df, non_real_el, unique_elements = utils.misc_preprocess(
            df, hdf5_file_species, global_min_radius, global_max_radius, 
            global_min_electronegativity, global_max_electronegativity,
            do_unique_elements=True
            )
    else:
        non_real_el = ['D', 'M', 'X']
        df = utils.misc_preprocess(
            df, hdf5_file_species, global_min_radius, global_max_radius, 
            global_min_electronegativity, global_max_electronegativity,
            do_unique_elements=False
            )
    print('Data preprocessed')
    print('Creating graphs...')
    graphs = []
    for index, row_data in tqdm(df.iterrows(), total=len(df)):
        data = utils.create_graph_object(
            row_data, index, hdf5_data, hdf5_species, non_real_el
            )
        graphs.append(data)
        if index > 2000:
            break
    df.reset_index(drop=True, inplace=True)
    print('Graphs created')
    print('Creating dataloaders...')
    user_input = input(
        "Please input batch_size (default 32): "
        )
    try:
        batch_size = int(user_input)
    except:
        print("Invalid input. Using default value of 32")
        batch_size = 32
    
    labels = [data.y.argmax() for data in graphs]
    train_graphs, temp_graphs, train_labels, temp_labels = train_test_split(
        graphs, labels, test_size=0.4, stratify=labels, random_state=42
    )
    
    val_graphs, test_graphs, val_labels, test_labels = train_test_split(
        temp_graphs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    train_dataset = GraphDataset(train_graphs, train_labels)
    val_dataset = GraphDataset(val_graphs, val_labels)
    test_dataset = GraphDataset(test_graphs, test_labels)
    user_input = input("Number of workers (default 4): ")
    try:
        num_workers = int(user_input)
    except:
        print("Invalid input. Using default value")
        num_workers = 4
        
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, 
        num_workers=num_workers, pin_memory=True
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, 
        num_workers=num_workers, pin_memory=True
        )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, 
        num_workers=num_workers, pin_memory=True)
    print('Dataloaders created')
    print('Creating model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(num_classes=7).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    test_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    train_precisions = []
    val_precisions = []
    test_precisions = []
    train_recalls = []
    val_recalls = []
    test_recalls = []
    train_f1s = []
    val_f1s = []
    test_f1s = []
    print('Model created')
    print('Training model...')
    user_input = input("Number of epochs (default 100): ")
    try:
        epochs = int(user_input)
    except:
        print("Invalid input. Using default value of 100")
        epochs = 100
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc,train_prec,train_rec,train_f1 = train_funcs.train(
            train_loader, model, device, optimizer, criterion
            )
        val_loss, val_acc, val_prec, val_rec, val_f1 = train_funcs.validate(
            val_loader, model, device, criterion
            )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        train_precisions.append(train_prec)
        val_precisions.append(val_prec)
        
        train_recalls.append(train_rec)
        val_recalls.append(val_rec)
        
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        print(f'Epoch {epoch+1}/{100}')
        print(\
            f'Train Loss: {train_loss:.4f}, \
            Accuracy: {train_acc:.4f}, \
            Precision: {train_prec:.4f}, \
            Recall: {train_rec:.4f}, \
            F1: {train_f1:.4f}'\
            )
        print(\
            f'Val Loss: {val_loss:.4f},\
            Accuracy: {val_acc:.4f},\
            Precision: {val_prec:.4f},\
            Recall: {val_rec:.4f},\
            F1: {val_f1:.4f}'\
            )
    user_input = input("Do you want to test the model? 1 (Yes) / 0 (No): ")
    try:
        test_model = bool(int(user_input))
    except:
        print("Invalid input. Using default value of no")
        test_model = False
    if test_model:
        test_loss, test_acc, test_prec, test_rec, test_f1 = train_funcs.test(
        test_loader, model, device, criterion
        )
        print(\
            f'Test Loss: {test_loss:.4f}, \
            Accuracy: {test_acc:.4f}, \
            Precision: {test_prec:.4f}, \
            Recall: {test_rec:.4f}, \
            F1: {test_f1:.4f}'\
            )
    user_input = input(
        "Do you want to plot loss/epoch graph? 1 (Yes) / 0 (No): "
        )
    try:
        plot_graph = bool(int(user_input))
    except:
        print("Invalid input. Using default value of yes")
        plot_graph = True
    if plot_graph:
        utils.plot_losses(train_losses, val_losses)