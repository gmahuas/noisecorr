################################################### LIBRARIES ###############################################################
#############################################################################################################################

# Dependencies
import os
import time
import datetime
import shutil
from importlib import reload
import sys
from tqdm import tqdm
import pickle as pkl

# Pytorch
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

# Some math and general purpose libraries
import numpy as np
import math
import scipy as sp
from scipy.signal import convolve as sc_convolve

# Plot libraries
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation

# Set random seeds
torch.manual_seed(2023)
np.random.seed(2023)

# Set GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

## Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the name of the current GPU
    device = 'cuda'
    current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print("Current GPU:", current_gpu_name)
else:
    device = 'cpu'
    print("No GPU available.")
    
#################################################### DATASETS ###############################################################
#############################################################################################################################

# Dataset for training and validation
class DatasetMultiCellStimulus(Dataset):
    
    """
    Dataset for decoder inference
    """
    
    def __init__(self, stimulus_frames, spike_counts):
        
        # Some fixed variables
        self.n_repeats = spike_counts.shape[-2]
        self.n_seq = spike_counts.shape[-1]
        
        # Data related variables
        self.spike_counts = spike_counts
        self.stimulus_frames = stimulus_frames

    def __len__(self):
        return self.n_repeats * self.n_seq
    
    def __getitem__(self, sample_idx):
        ## We include the current time bin.......
        # Get spike counts
        sample_spike_counts = self.spike_counts[:,
                                                :,
                                                sample_idx % self.n_repeats,
                                                sample_idx // self.n_repeats].flatten()
        
        ##......to predict the current frame
        # Get stimulus
        sample_stimulus_frame = self.stimulus_frames[:,
                                                     :,
                                                     sample_idx // self.n_repeats].flatten()
        
        # Transforming into pytorch tensors
        sample_spike_counts = torch.tensor(sample_spike_counts, dtype=torch.float32)
        sample_stimulus_frame = torch.tensor(sample_stimulus_frame, dtype=torch.float32)
        
        return sample_stimulus_frame, sample_spike_counts
    
##################################################### MODELS ################################################################
#############################################################################################################################

# Linear decoder
class LinearDecoder(nn.Module):
    def __init__(self, n_cells, n_bins_past, n_x, n_y, dropout_prob=0):
        super(LinearDecoder, self).__init__()
        
        self.flatten = torch.nn.Flatten()
        self.linear_layer = torch.nn.Linear((n_cells+1)*n_bins_past, n_x*n_y, bias=True)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.nonlinearity = torch.nn.functional.sigmoid
        
    def forward(self, spikes):

        # Linear decoder
        out = self.flatten(spikes)
        out = self.linear_layer(out)
        out = self.dropout_layer(out)
              
        # Nonlinear output
        out = self.nonlinearity(out)
        
        return out
    
################################################# REGULARIZATION ############################################################
#############################################################################################################################

def l1_penalty(model, factor=1e-4):
    l1_loss = 0
    for param_name, param in model.named_parameters():
        if 'linear_layer' in param_name:  # Apply L1 penalty only to the linear_layer
            l1_loss += torch.norm(param, 1)
    return factor * l1_loss

def laplacian_penalty(model, factor=1e-4):
    # Define the Laplacian kernel
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32)
    
    laplacian_kernel = laplacian_kernel.to(device)

    # Compute the 2D Laplacian across the first two axes
    laplacian_loss = 0
    for param_name, param in model.named_parameters():
        if 'linear_layer' in param_name:
            param = param.reshape(n_x, n_y, -1)
            for i in range(param.shape[2]):
                laplacian_loss += (F.conv2d(param[:, :, i].unsqueeze(0), laplacian_kernel.unsqueeze(0).unsqueeze(0), padding=1)**2).sum()

    return factor*laplacian_loss

#################################################### FOLDER MANAGEMENT ######################################################
#############################################################################################################################

# Read from .txt what is the model to simulate
with open('input.txt', 'r') as file:
    i_folder = int(file.readline().strip())

# Open the file in write mode to update the value
with open('input.txt', 'w') as file:
    file.write(str(i_folder+1))

# Get model folders list
def read_folder_names_from_txt(filename):
    folder_names = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            folder_name = line.strip()
            folder_names.append(folder_name)
    return folder_names

folder_names = read_folder_names_from_txt("../models/folder_names.txt")

# Folder of the model to train
model_folder = f'{folder_names[i_folder]}'

print("Model to train:", model_folder)

# Create folder
if not os.path.exists('./models'):
    os.makedirs('./models')

######################################################## LOAD DATA ##########################################################
#############################################################################################################################

# Data parameters
n_rep = 5000
n_cells = 49

data_file = f'spikes_sim_{n_rep}rep_{n_cells}cells_{model_folder}.npy'

# Load stimulus
with open(f'../simulate/spikes_data_train/stimulus_data_{n_cells}cells.pkl', 'rb') as file:
    stimulus_frames, check_size_list, xphase_list, yphase_list = pkl.load(file).values()

stimulus_frames = stimulus_frames.astype('float32')[:,:,1:]
check_size_list = np.array(check_size_list[1:])
n_x, n_y = stimulus_frames.shape[:2]

## Load spikes
spikes = np.load('../simulate/spikes_data_train/'+data_file)
n_cells, n_t, n_rep, n_seq = spikes.shape

# Shuffle spike trains
spikes_shuffled = spikes.copy()
for i_cell in range(n_cells):
    spikes_shuffled[i_cell,:,:,:] = np.transpose(spikes_shuffled[i_cell,:,np.random.permutation(n_rep),:], (1,0,2))

## Prepare datasets
# Training and validation set
reps = np.random.permutation(n_rep)
n_reps_train = 3*n_rep//4
train_reps = reps[:n_reps_train]
n_reps_val = n_rep-n_reps_train
val_reps = reps[n_reps_train:]

# Time points to keep
t_onset = 0 # we already dropped the first five bins (after simulation)
n_bins_past = 5 # bins in the past to consider
n_t_train = n_bins_past
n_t_val = n_bins_past

# Dependent data
# Training
data_train = np.concatenate((spikes[:,t_onset:t_onset+n_bins_past,train_reps,:], np.ones((n_t_train,n_reps_train,n_seq))[None,...]), axis=0)
# Validation
data_val = np.concatenate((spikes[:,t_onset:t_onset+n_bins_past,val_reps,:], np.ones((n_t_val,n_reps_val,n_seq))[None,...]), axis=0)

# Shuffled data
# Training
data_shuffled_train = np.concatenate((spikes_shuffled[:,t_onset:t_onset+n_bins_past,train_reps,:], np.ones((n_t_train,n_reps_train,n_seq))[None,...]), axis=0)
# Validation
data_shuffled_val = np.concatenate((spikes_shuffled[:,t_onset:t_onset+n_bins_past,val_reps,:], np.ones((n_t_val,n_reps_val,n_seq))[None,...]), axis=0)

##################################################### TRAIN DECODERS ########################################################
#############################################################################################################################

# Training parameters
batch_size = 2048*4
learning_rate = 2
num_epochs = 200
patience = 6
min_improvement = 5e-5
workers = 8

# Regularization
# lambda_l1 = 0
# lambda_lapl = 0

# Model and data
model_types = ['lin']
data_types = ['indep', 'dep']

# Loop over training
for model_type in model_types:
    for data_type in data_types:
        
        # Create model
        model = LinearDecoder(n_cells, n_bins_past, n_x, n_y)
        model = model.to(device)
        
        # Loss function
        criterion = torch.nn.BCELoss()

        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, factor = 0.25)

        # Lists to store losses
        train_losses = []
        val_losses = []

        # Early stopping
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state_dict = None

        # Create dataset
        if data_type == 'dep':
            # Unshuffled data
            # Training set
            DatasetTrain = DatasetMultiCellStimulus(stimulus_frames, data_train)
            TrainLoader = DataLoader(DatasetTrain, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = workers)

            # Validation set
            DatasetVal = DatasetMultiCellStimulus(stimulus_frames, data_val)
            ValLoader = DataLoader(DatasetVal, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = workers)

        elif data_type == 'indep':
            # Shuffled data
            # Training set
            DatasetTrain = DatasetMultiCellStimulus(stimulus_frames, data_shuffled_train)
            TrainLoader = DataLoader(DatasetTrain, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = workers)

            # Validation set
            DatasetVal = DatasetMultiCellStimulus(stimulus_frames, data_shuffled_val)
            ValLoader = DataLoader(DatasetVal, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = workers)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for batch_idx, (stimulus_batch, spikes_batch) in enumerate(TrainLoader):
                print(f'Training - Epoch [{epoch+1}/{num_epochs}], Batch #{batch_idx+1}/{len(TrainLoader)}', end='\r')
                stimulus_batch = stimulus_batch.to(device)
                spikes_batch = spikes_batch.to(device)

                # Forward pass
                outputs = model(spikes_batch)

                # Calculate the loss
                loss = criterion(outputs, stimulus_batch)
                # loss = criterion(outputs, stimulus_batch) + l1_penalty(model, factor=lambda_l1) + laplacian_penalty(model, factor=lambda_lapl)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(TrainLoader)
            train_losses.append(avg_loss)
            print(f'Training - Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}                               ')

            # Validation loop
            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                for val_batch_idx, (val_stimulus_batch, val_spikes_batch) in enumerate(ValLoader):
                    print(f'Validation - Epoch [{epoch+1}/{num_epochs}], Batch #{val_batch_idx+1}/{len(ValLoader)}', end='\r')
                    val_stimulus_batch = val_stimulus_batch.to(device)
                    val_spikes_batch = val_spikes_batch.to(device)

                    val_outputs = model(val_spikes_batch)

                    val_loss = criterion(val_outputs, val_stimulus_batch)

                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(ValLoader)
                
                # Step the learning rate scheduler based on validation loss
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(avg_val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < old_lr:
                    print("Learning rate has decreased: LR = ", new_lr)

                val_losses.append(avg_val_loss)
                print(f'Validation - Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_val_loss:.4f}')

                # Save the best model
                if avg_val_loss <= best_val_loss-min_improvement:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0
                    best_model_state_dict = model.state_dict()
                    best_model_path = os.path.join('./models', f'best_{model_type}_model_{data_type}_{model_folder}.pth')
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(best_model_state_dict, best_model_path)
                else:
                    early_stop_counter += 1
                    print(f'Validation loss not improving: {patience-early_stop_counter} steps before early stopping.')

                if early_stop_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1} as validation loss did not improve.')
                    break

        print('Training finished.')
        
        # Plot loss
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.title(f'{data_type} data, {model_type} model')
        plt.savefig(f'./models/losses_{data_type}_{model_type}_{model_folder}.png')
        
