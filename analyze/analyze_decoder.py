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
import scipy.fftpack as sfft
from scipy.fftpack import dct
import scipy.signal
from scipy.linalg import hadamard

# Plot libraries
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation

# Set random seeds
torch.manual_seed(2023)
np.random.seed(2023)

# Set GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

# Check if CUDA (GPU support) is available
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

## Full model stimulus dataset
class DatasetDecoder(Dataset):
    
    """
    Dataset for decoder testing
    """
    
    def __init__(self, spike_counts):
        
        # Some fixed variables
        self.n_repeats = spike_counts.shape[2]
        
        # Data related variables
        self.spike_counts = spike_counts
        
    def __len__(self):
        return self.n_repeats
    
    def __getitem__(self, sample_idx):
        
        # Get spike counts
        sample_spike_counts = self.spike_counts[:,
                                                :,
                                                sample_idx].flatten()
        
        # Transforming into pytorch tensors
        sample_spike_counts = torch.tensor(sample_spike_counts, dtype=torch.float32)
        
        return sample_spike_counts
    
##################################################### MODELS ################################################################
#############################################################################################################################

# Linear decoder
class LinearDecoder(nn.Module):
    def __init__(self, n_cells, n_bins_past, n_x, n_y):
        super(LinearDecoder, self).__init__()
        
        self.flatten = torch.nn.Flatten()
        self.linear_layer = torch.nn.Linear((n_cells+1)*n_bins_past, n_x*n_y, bias=True)
        self.nonlinearity = torch.nn.functional.sigmoid
        
    def forward(self, spikes):

        # Linear decoder
        out = self.flatten(spikes)
        out = self.linear_layer(out)
              
        # Nonlinear output
        out = self.nonlinearity(out)
        
        return out
    
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

# Folder of the model to test
model_folder = f'{folder_names[i_folder]}'

print("Model to test:", model_folder)

# Create folder
if not os.path.exists('./information'):
    os.makedirs('./information')
    
######################################################## LOAD DATA ##########################################################
#############################################################################################################################
    
# Data parameters
n_rep = 5000
n_cells = 49

data_file = f'spikes_sim_{n_rep}rep_{n_cells}cells_{model_folder}.npy'

# Load stimulus
with open(f'../simulate/spikes_data_test/stimulus_data_{n_cells}cells.pkl', 'rb') as file:
    stimulus_frames, check_size_list, xphase_list, yphase_list = pkl.load(file).values()

stimulus_frames = stimulus_frames.astype('float32')[:,:,1:]
check_size_list = np.array(check_size_list)
n_x, n_y = stimulus_frames.shape[:2]

## Load spikes
spikes = np.load('../simulate/spikes_data_test/'+data_file)
n_cells, n_t, n_rep, n_seq = spikes.shape

# Shuffle spike trains
spikes_shuffled = spikes.copy()
for i_cell in range(n_cells):
    spikes_shuffled[i_cell,:,:,:] = np.transpose(spikes_shuffled[i_cell,:,np.random.permutation(n_rep),:], (1,0,2))
    
##################################################### LOAD DECODERS #########################################################
#############################################################################################################################

# Stimulus shape
n_x, n_y = stimulus_frames.shape[:2]

# Spike hisory to integrate
n_bins_past = 5

# Linear model
parameters = {
    "n_cells": n_cells,
    "n_bins_past": n_bins_past,
    "n_x": n_x,
    "n_y": n_y
}
model = LinearDecoder

# Models folder
models_folder = '../train_decoder/models'
print('models located in folder: '+models_folder)

# Models type
model_type = 'lin'

## Dep model
dep_model = model(**parameters)
saved_model_path = f'./{models_folder}/best_{model_type}_model_dep_{model_folder}.pth'
dep_model.load_state_dict(torch.load(saved_model_path))
dep_model = dep_model.to(device)
dep_model.eval();

## Indep model
indep_model = model(**parameters)
saved_model_path = f'./{models_folder}/best_{model_type}_model_indep_{model_folder}.pth'
indep_model.load_state_dict(torch.load(saved_model_path))
indep_model = indep_model.to(device)
indep_model.eval();

#################################################### PREPARE DATASETS #######################################################
#############################################################################################################################

# Time points to keep
n_reps_test = n_rep
t_onset = 0 # we already dropped the first five bins (after simulation)
n_t_test = n_t-t_onset

# Dependent data
# Test
data_test = np.concatenate((spikes[:,t_onset:,:n_reps_test,:n_seq], np.ones((n_t_test,n_reps_test,n_seq))[None,:]), axis=0)

# Shuffled data
# Test
data_shuffled_test = np.concatenate((spikes_shuffled[:,t_onset:,:n_reps_test,:n_seq], np.ones((n_t_test,n_reps_test,n_seq))[None,:]), axis=0)

################################################### FIND CENTRAL ZONE #######################################################
#############################################################################################################################
# Position of the central cell for the independent decoder
# Recover weights
weights = indep_model.linear_layer.weight.data.cpu().numpy()
weights = weights.reshape(n_x, n_y, -1)
weights = weights.reshape(n_x, n_y, n_cells+1, n_bins_past)

i_central_cell = 24
RF = weights[:,:,i_central_cell,:].var(-1)
center_pos = np.argmax(RF)//n_y, np.argmax(RF)%n_y

# Find Voronoi/Brillouin cell
def hexagon_vertices(center, radius):
    """Calculate hexagon vertices given the center (x, y) and radius of the circumscribed circle."""
    theta_offset = np.pi / 6  # offset to align one vertex at the right (0 degrees)
    angles = theta_offset + np.linspace(0, 2 * np.pi, 7)[:-1]  # angles for vertices
    x, y = center
    return [(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]

def is_point_in_hexagon(point, vertices):
    """Use the ray-casting algorithm to determine if the point is in the hexagon."""
    x, y = point
    count = 0
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        if (y0 <= y <= y1) or (y1 <= y <= y0):
            x_edge = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < x_edge:
                count += 1
    return count % 2 == 1

def find_points_in_hexagon(array_shape, center, radius):
    """Return a list of points within a hexagon for a given 2D array shape."""
    nx, ny = array_shape
    hex_vertices = hexagon_vertices(center, radius)
    points_in_hexagon = []
    for ix in range(nx):
        for iy in range(ny):
            if is_point_in_hexagon((ix, iy), hex_vertices):
                points_in_hexagon.append((ix, iy))
    return points_in_hexagon

center = (center_pos[0], center_pos[1])  # Center coordinates of the hexagon
radius = 3 # Half of the interneuron distance
array_shape = (n_x, n_y)
points_brillouin = np.array(find_points_in_hexagon(array_shape, center, radius))
n_points_brillouin = points_brillouin.shape[0]

################################################ DECODE IN CENTRAL ZONE #####################################################
#############################################################################################################################

def decode(model_to_use, dataset_to_use):
    predictions = []
    for batch_idx, spikes_batch in enumerate(dataset_to_use):
        spikes_batch = spikes_batch.to(device)

        # Decode
        pred = model_to_use(spikes_batch).detach().cpu().numpy().astype('float16')

        # Store
        predictions += [pred]
        
    # Reshape predictions
    predictions = np.concatenate(predictions)
    predictions = predictions.reshape(n_reps_test, (n_t_test-2*n_bins_past), n_x, n_y)
    
    return predictions

batch_size = 1024

# Storages
predictions_indep = np.zeros((n_seq, n_reps_test, n_points_brillouin), dtype='float16')
predictions_dep = np.zeros((n_seq, n_reps_test, n_points_brillouin), dtype='float16')

for i_seq in tqdm(range(n_seq)):
    # Unshuffled data - dependent decoder
    torch.cuda.empty_cache()
    DatasetTestUnshuffled = DatasetDecoder(data_test[:,:,:,i_seq])
    TestLoaderUnshuffled = DataLoader(DatasetTestUnshuffled, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 1)
    predictions_dep_temp = decode(dep_model, TestLoaderUnshuffled).squeeze()
    predictions_dep[i_seq,...] = predictions_dep_temp[:,points_brillouin[:,0], points_brillouin[:,1]]
    
    # Shuffled data - independent decoder
    torch.cuda.empty_cache()
    DatasetTestShuffled = DatasetDecoder(data_shuffled_test[:,:,:,i_seq])
    TestLoaderShuffled = DataLoader(DatasetTestShuffled, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 1)
    predictions_indep_temp = decode(indep_model, TestLoaderUnshuffled).squeeze()
    predictions_indep[i_seq,...] = predictions_indep_temp[:,points_brillouin[:,0], points_brillouin[:,1]]
    
predictions_indep = predictions_indep[:,:,:,None]
predictions_dep = predictions_dep[:,:,:,None]

## Reconstruct true stimulus
true_stimulus = np.transpose(stimulus_frames, (2,0,1))[:,points_brillouin[:,0], points_brillouin[:,1]]
true_stimulus = true_stimulus[:,:,None]

# Save decoded data
# np.save(f'./decoded_stim/predictions_dep_true_{n_cells}cells_{model_folder}.npy', predictions_dep)
# np.save(f'./decoded_stim/predictions_indep_true_{n_cells}cells_{model_folder}.npy', predictions_indep)

################################################## COMPUTE INFORMATION ######################################################
#############################################################################################################################

def compute_information(check_size_list, size, prediction, n_bins_hist, true_stimulus):
    n_rep = prediction.shape[1]
    
    # Find trial indices corresponding to size
    size_indices = np.where(np.array(check_size_list) == size)[0]

    # Recover predicted color in trials corresponding to size
    prediction_per_size = prediction[size_indices,:]
    
    # True pixel color in trials corresponding to size
    true_color_pixel = true_stimulus[size_indices]

    # Compute conditional histograms (conditioned on true pixel color)
    bin_range = np.linspace(0,1,n_bins_hist+1)
    hist_cond = np.zeros((2, n_bins_hist))
    for i_color in range(2):
        color_pixel_indices = np.where(true_color_pixel==i_color)[0]
        hist_cond[i_color,:] = np.histogram(prediction_per_size[color_pixel_indices,:].flatten(), bin_range)[0]/(n_rep*len(color_pixel_indices))
        
    # True pixel color probabilities
    p_color = np.zeros((2))
    p_color[0] = np.sum(true_color_pixel==0)/len(true_color_pixel)
    p_color[1] = np.sum(true_color_pixel==1)/len(true_color_pixel)
    
    # Compute marginal histogram
    hist_marg = np.sum(hist_cond*p_color[:,None], 0)

    ## Compute entropies
    eps = 1e-24

    H_cond = -np.sum(hist_cond*np.log(hist_cond+eps), 1)
    H_marg = -np.sum(hist_marg*np.log(hist_marg+eps))

    ## Compute information
    I = H_marg - np.sum(H_cond*p_color)
    
    return I

# Parameters
n_bins = 20 # for histogram
n_sizes = len(np.unique(check_size_list))

# Calculation loop
I_indep_per_px = np.zeros((n_points_brillouin, 1, n_sizes))
I_dep_per_px = np.zeros((n_points_brillouin, 1, n_sizes))

for x in tqdm(range(n_points_brillouin)):
    for y in range(1):
        predictions_indep_xy = predictions_indep[:,:,x,y]
        predictions_dep_xy = predictions_dep[:,:,x,y]
        true_stimulus_xy = true_stimulus[:,x,y]
        for i_size, size in enumerate(np.unique(check_size_list)):
            I_indep_per_px[x,y,i_size] = compute_information(check_size_list, size, predictions_indep_xy, n_bins, true_stimulus_xy)
            I_dep_per_px[x,y,i_size] = compute_information(check_size_list, size, predictions_dep_xy, n_bins, true_stimulus_xy)
            
## Save information
np.save(f'./information/information_indep_{model_folder}.npy', I_indep_per_px)
np.save(f'./information/information_dep_{model_folder}.npy', I_dep_per_px)
