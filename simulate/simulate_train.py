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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import Tensor
from torch import nn
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

# Some math and general purpose libraries
import numpy as np
import math
import scipy as sp
from scipy.signal import convolve as sc_convolve
from scipy.linalg import hadamard

# Plot libraries
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.path import Path
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
    
##################################################### MODELS ################################################################
#############################################################################################################################

# Raised cosine basis for the stimulus and the couplings
def raised_cosine_basis(first_peak, last_peak, stretch, n_basis, nt_integ):

    # Natural logarithm functions
    eps = 1e-16
    def nl(x):
        return np.log(eps + x)

    def inverse_nl(x):
        return np.exp(x)

    nlpos = [nl(first_peak + stretch), nl(last_peak + stretch)]
    peaks = np.expand_dims(np.linspace(nlpos[0], nlpos[1], n_basis), axis=1)
    spacing = peaks[1] - peaks[0]

    timepoints = np.expand_dims(np.arange(int(inverse_nl(nlpos[1] + 5 * spacing) - stretch)), axis=0)
    nt = timepoints.shape[0]

    def raised_cosine(x, c, dc):
        return (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x - c) * np.pi / dc / 2))) + 1) / 2

    basis = raised_cosine(np.tile(nl(timepoints + stretch), (n_basis, 1)), np.tile(peaks, (1, nt)), spacing)
    basis = basis[:, :nt_integ] / np.expand_dims(np.sum(basis, axis=1), axis=1)
    basis = basis[:, ::-1]
    
    return basis

# Custom separable convolution layer
class Conv2Plus1D(nn.Sequential):
    """
    3D Convolutional layer with space - time separability
    """
    def __init__(self, in_planes: int = 1, space_per_input_filters: int = 2, time_per_space_filter: int = 1,
                 kernel_time: int = 3, kernel_space: int = 3,
                 bias: bool = True) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes,
                in_planes * space_per_input_filters,
                kernel_size=(1, kernel_space, kernel_space),
                stride=(1, 1, 1),
                padding=(0, kernel_space // 2, kernel_space // 2),
                bias=bias,
                groups=in_planes,
            ),
            nn.ELU(inplace=True),
            nn.Conv3d(
                in_planes * space_per_input_filters,
                in_planes * space_per_input_filters * time_per_space_filter,
                kernel_size=(kernel_time, 1, 1),
                stride=(1, 1, 1),
                padding=(kernel_time // 2, 0, 0),
                bias=bias,
                groups=in_planes * space_per_input_filters),
        )
        
# Single cell model for the simulation
class StimulusModel_single_cell(nn.Module):
    """
    Space-time separable convolutional neural network with factorized readout layer
    """
    def __init__(self, input_channels: int = 1,
                 space_filters_1: int = 2, time_per_space_filter_1: int = 1,
                 kernel_time_1: int = 3, kernel_space_1: int = 3,
                 pooling_time_1: int = 8, pooling_space_1: int = 12,
                 space_filters_2: int = 2, time_per_space_filter_2: int = 1,
                 kernel_time_2: int = 3, kernel_space_2: int = 3,
                 pooling_time_2: int = 4, pooling_space_2: int = 6,
                 bias: bool = True,
                 device: str = "cpu"):
        super().__init__()
        
        # Single cell model
        num_cells = 1
        
        # First convolution
        self.conv1 = Conv2Plus1D(input_channels, space_filters_1, time_per_space_filter_1, kernel_time_1, kernel_space_1, bias=bias)
        output_first_conv = input_channels * space_filters_1 * time_per_space_filter_1
        self.bn1 = nn.BatchNorm3d(output_first_conv)
        self.nl1 = nn.ELU(inplace=True)
        self.avgpool1 = nn.AdaptiveAvgPool3d((pooling_time_1, pooling_space_1, pooling_space_1))
        
        # Second convolution
        self.conv2 = Conv2Plus1D(output_first_conv, space_filters_2, time_per_space_filter_2, kernel_time_2, kernel_space_2, bias=bias)
        output_second_conv = output_first_conv * space_filters_2 * time_per_space_filter_2
        self.bn2 = nn.BatchNorm3d(output_second_conv)
        self.nl2 = nn.ELU(inplace=True)
        self.avgpool2 = nn.AdaptiveAvgPool3d((pooling_time_2, pooling_space_2, pooling_space_2))
        
        # Readout
        # Define the projection tensors
        self.space_time_readout = nn.Parameter(torch.empty(1, pooling_time_2, pooling_space_2, pooling_space_2, num_cells))
        self.features_readout = nn.Parameter(torch.empty(output_second_conv, num_cells))
        # Inititalize the tensors
        torch.nn.init.kaiming_normal_(self.space_time_readout)
        torch.nn.init.kaiming_normal_(self.features_readout)
        # Define the nonlinearity
        self.nl3 = nn.ELU(inplace=True)
        
        # Output nonlinearity
        self.output_nonlinearity_bias = nn.Parameter(torch.zeros(1, num_cells))

    def forward(self, stimulus_input: Tensor) -> Tensor:
        
        # First convolution
        out = self.conv1(stimulus_input)
        out = self.bn1(out)
        out = self.nl1(out)
        out = self.avgpool1(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nl2(out)
        out = self.avgpool2(out)

        # Readout
        out = (out[...,None]*self.space_time_readout).sum(dim = (2, 3, 4))
        out = self.nl3(out)
        out = (out*self.features_readout).sum(dim = 1)
        
        # Output nonlinearity
        out = out + self.output_nonlinearity_bias
        
        return out
    
# Code that loads the parameters corresponding to a selected cell from a previously trained population model
def load_specific_cell_parameters(trained_model_path: str, i_cell: int, parameters: dict, device: str) -> StimulusModel_single_cell:
    # Load the trained model state dictionary
    state_dict = torch.load(trained_model_path, map_location=device)

    # Create a new single cell model with num_cells set to 1
    new_model = StimulusModel_single_cell(**parameters)

    # Extract parameters for the specific cell index
    space_time_readout = state_dict['space_time_readout'][:, :, :, :, i_cell:i_cell+1]
    features_readout = state_dict['features_readout'][:, i_cell:i_cell+1]
    output_nonlinearity_bias = state_dict['output_nonlinearity_bias'][:, i_cell:i_cell+1]

    # Update the new model's parameters with the extracted ones
    new_model.space_time_readout.data = space_time_readout
    new_model.features_readout.data = features_readout
    new_model.output_nonlinearity_bias.data = output_nonlinearity_bias

    # Copy over the remaining parameters (conv layers, bn layers, etc.)
    for name, param in state_dict.items():
        if 'space_time_readout' not in name and 'features_readout' not in name and 'output_nonlinearity_bias' not in name:
            new_model.state_dict()[name].copy_(param)

    return new_model
    
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

# Folder of the model to simulates
model_folder = f'{folder_names[i_folder]}'

print("Model to simulate:", model_folder)

# Folder to save the spikes
spikes_folder = './spikes_data_train'

# Create folder
if not os.path.exists(spikes_folder):
    os.makedirs(spikes_folder)

################################################ STIMULUS & POPULATION ######################################################
#############################################################################################################################

# Stimulus and simulation temporal paramters
stimulus_framerate = 4 # original stimulus framerate
bin_size = 1e-3 # bin size to use for the model

# Size original stimulus
n_x_original, n_y_original = 24, 24

# Size synthetic stimulus (padding for the large synthetic population)
n_x, n_y = n_x_original+38, n_y_original+32

# Bias to account for the padding of the original stimulus
biasx = (n_x-n_x_original)//2
biasy = (n_y-n_y_original)//2

# Positions of the original cells
file_path = './contours.pkl'
with open(file_path, 'rb') as f:
    contours = pkl.load(f)
    
subset_cells = [221, 226, 247, 251, 254, 259, 271]

centers_list = []
for cell in subset_cells:
    i_sta = np.where(np.array(contours[1])==cell)[0][0]
    centers_list += [(contours[0][i_sta][:,1].mean()+biasx, contours[0][i_sta][:,0].mean()+biasy)]
    
# Position of the central cell
central_cell_id = 259
central_sta_id = np.where(np.array(contours[1])==central_cell_id)[0][0]
central_cell_center = (contours[0][central_sta_id][:,1].mean()+biasx), (contours[0][central_sta_id][:,0].mean()+biasy)
    
# Lattice parameters
neighb_cells = np.setdiff1d(np.arange(7), 5)
mean_distance_neighbors = (np.sum((np.array(centers_list)[neighb_cells,:]-np.array(centers_list)[5,:])**2, 1)**0.5).mean()
mean_distance_neighbors = np.round(mean_distance_neighbors).astype(dtype='int')

# Bias to center the RF of the central cell on the lattice points during simulations
biasx_RF = +1
biasy_RF = -1

# Stimulus and synthetic lattice
def generate_square_lattice(N_x, N_y, d, x_center, y_center):
    x, y = np.meshgrid(np.arange(N_x) * d, np.arange(N_y) * d)
    x, y = x.flatten(), y.flatten()
    coords = np.vstack((x, y))
    coords_center = coords[:, N_x * (N_y // 2) + N_x//2]
    coords[0, :] += x_center - coords_center[0]
    coords[1, :] += y_center - coords_center[1]
    return coords[0, :], coords[1, :]

def generate_triangular_lattice_population(N_x, N_y, d, x_center, y_center):
    # Create the triangular lattice
    x_coords = np.arange(N_x) * d
    y_coords = np.arange(N_y) * d * np.sqrt(3) / 2
    offset = (d / 2) * (np.arange(N_y) % 2)
    x_coords = np.repeat(x_coords, N_y) + np.tile(offset, N_x)
    y_coords = np.tile(y_coords, N_x)
    
    # Centering the lattice
    mean_x, mean_y = x_coords[N_x * (N_y // 2) + N_x//2], y_coords[N_x * (N_y // 2) + N_x//2]
    x_coords += (x_center - mean_x)
    y_coords += (y_center - mean_y)
    
    # Create the coordinate array
    coords = np.vstack([x_coords, y_coords])
    
    return coords[0, :], coords[1, :]

def generate_color_pattern(N):
    # Pattern for a classical checkerboard
    color_pattern = []
    for i in range(N):
        for j in range(N):
            color_pattern.append((i + j) % 2)
    return color_pattern

# def generate_color_pattern_stripes(N):
#     # Pattern for gratings
#     color_pattern = []
#     for i in range(N):
#         for j in range(N):
#             color_pattern.append(j % 2)
#     return color_pattern

# Generate the population lattice
N_x, N_y = 7, 7  # Number of cells in each dimension
n_cells_synth = N_x*N_y
x_center, y_center = n_x//2, n_y//2
x_coords, y_coords = generate_triangular_lattice_population(N_x, N_y, mean_distance_neighbors, x_center, y_center)
population_positions = np.array((x_coords, y_coords))

# Generate the stimulus
unique_check_size_list = [5,6,7,8,9,10,11,12,13,14,15]

dx, dy = 2, 2
dL_x, dL_y = 4, 4
unique_xphases_list = np.arange(-dL_x,dL_x+dx, dx)
unique_yphases_list = np.arange(-dL_y,dL_y+dy, dy)

n_sizes = len(unique_check_size_list)
frames_list = [np.ones((n_x, n_y, 1))]
check_size_list = []
xphases_list = []
yphases_list = []
polarity_list = []

for check_size in unique_check_size_list:
    for i_x_phase, x_phase in enumerate(unique_xphases_list):
        for i_y_phase, y_phase in enumerate(unique_yphases_list):
            for polarity in range(2):
                N_checks = 31
                x_coords_checks, y_coords_checks = generate_square_lattice(N_checks, N_checks, check_size, x_center, y_center)

                # Color patterns for the checks
                color_pattern = generate_color_pattern(N_checks)

                # Create stimulus with custom color pattern
                frame = np.ones((n_y, n_x))
                for i_check in range(N_checks**2):
                    x_center_check = x_coords_checks[i_check] + x_phase
                    y_center_check = y_coords_checks[i_check] + y_phase

                    x_coords_mesh, y_coords_mesh = np.meshgrid(np.arange(n_x), np.arange(n_y))
                    mask = ((y_coords_mesh >= (y_center_check - check_size / 2)) &
                            (y_coords_mesh <= (y_center_check + check_size / 2)) &
                            (x_coords_mesh >= (x_center_check - check_size / 2)) &
                            (x_coords_mesh <= (x_center_check + check_size / 2)))

                    frame[mask] = polarity*color_pattern[i_check] + (1-polarity)*(1-color_pattern[i_check])

                # Append to frames_list
                frames_list += [frame.T[:,:,None]]
                
                # Store parameters of the frame
                check_size_list += [check_size]
                xphases_list += [x_phase]
                yphases_list += [y_phase]
                polarity_list += [polarity]
            
frames_list = np.concatenate(frames_list, axis=-1)

# Save stimulus data
data_dict = {}
data_dict['stimulus_frames'] = frames_list
data_dict['check_size_list'] = check_size_list
data_dict['xphases_list'] = xphases_list
data_dict['yphases_list'] = yphases_list

with open(f'./{spikes_folder}/stimulus_data_{n_cells_synth}cells.pkl', 'wb') as file:
    pkl.dump(data_dict, file)

# Reconstruct couplings
# Interaction and raised cosine bases parameters
int_past_integration = 0.05
n_bins_past_int = np.floor(int_past_integration/bin_size).astype('int')
coupl_first_peak, coupl_last_peak, coupl_stretch, coupl_n_basis = 0, 25, 5, 7
self_first_peak, self_last_peak, self_stretch, self_n_basis = 0, 40, 5, 7

# Reconstruct bases
basis_coupl = raised_cosine_basis(first_peak=coupl_first_peak,
                                  last_peak=coupl_last_peak,
                                  stretch=coupl_stretch,
                                  n_basis=coupl_n_basis,
                                  nt_integ=n_bins_past_int)

basis_self = raised_cosine_basis(first_peak=self_first_peak,
                                 last_peak=self_last_peak,
                                 stretch=self_stretch,
                                 n_basis=self_n_basis,
                                 nt_integ=n_bins_past_int)

# Load parameters
model_state_dict = torch.load(f'../models/{model_folder}/best_model_population.pth')
self_filters_weights = model_state_dict['self_filter.weight'].cpu().numpy()
coupl_filters_weights = model_state_dict['coupl_filters.weight'].cpu().numpy()

n_cells = self_filters_weights.shape[0]

# Reconstruct the self coupling
self_reconstruction = (self_filters_weights@basis_self).squeeze()

# Reconstruct the couplings
coupl_reconstruction = (coupl_filters_weights@basis_coupl).squeeze(1)

# Reassemble the filters cell-wise
filters_reconstruction = np.zeros((n_cells, n_cells, n_bins_past_int), dtype='float32')
for i_cell in range(n_cells):
    # Self filters
    filters_reconstruction[i_cell, i_cell, :] = self_reconstruction[i_cell,:]
    # Coupling filters
    for j_cell_idx, j_cell in enumerate(np.setdiff1d(np.arange(n_cells), i_cell)):
        filters_reconstruction[i_cell, j_cell, :] = coupl_reconstruction[i_cell, j_cell_idx,:]
        
# Mean coupling filters between the central cell and its neighbors
i_cell = 5
out_of_diag = np.setdiff1d(np.arange(n_cells),i_cell)
coupling_filter = (filters_reconstruction[i_cell,out_of_diag,:].mean(0)+filters_reconstruction[out_of_diag,i_cell,:].mean(0))/2

# Self coupling (or history) filter
history_filter = filters_reconstruction[i_cell,i_cell,:]

# Connectivity matrix for the synhetic population
distance_matrix = np.sum((population_positions[:,None,:]-population_positions[:,:,None])**2, 0)**0.5
thres = mean_distance_neighbors + 1e-2
connectivity_matrix = distance_matrix<=thres
connectivity_matrix = connectivity_matrix - np.eye(n_cells_synth)

# Stimulus raised cosine basis
# Stimulus and raised cosine basis parameters
past_integration = 0.8
n_bins_past_stim = np.floor(past_integration/bin_size).astype('int')
stim_first_peak, stim_last_peak, stim_stretch, stim_n_basis = int(0.*n_bins_past_stim), int(0.9*n_bins_past_stim), 0.2/bin_size, 8
basis_stim = raised_cosine_basis(first_peak=stim_first_peak,
                                 last_peak=stim_last_peak,
                                 stretch=stim_stretch,
                                 n_basis=stim_n_basis,
                                 nt_integ=n_bins_past_stim).astype('float32')

# Push raised cosine bases to GPU
basis_stim = torch.from_numpy(basis_stim)
basis_stim = basis_stim.to(device)

# Stimulus model parameters
parameters = {
    'input_channels': 1,
    'space_filters_1': 4,
    'time_per_space_filter_1': 2,
    'kernel_time_1': 3,
    'kernel_space_1': 3,
    'pooling_time_1': 6,
    'pooling_space_1': 12,
    'space_filters_2': 2,
    'time_per_space_filter_2': 1,
    'kernel_time_2': 3,
    'kernel_space_2': 3,
    'pooling_time_2': 3,
    'pooling_space_2': 8,
    'bias': True,
    'device': device
}

# Load stimulus model
# Path to the saved full population model
path_save = f'../models/{model_folder}/stimulus_model.pth'

# Load the single cell model
i_cell = 5
single_cell_model = load_specific_cell_parameters(path_save, i_cell, parameters, device=device)
single_cell_model.eval()
single_cell_model = single_cell_model.to(device)

################################################## STIMULUS PROCESSING ######################################################
#############################################################################################################################

# Stimulus snippets parameters
# Frame duration
flash_duration = 0.25

# Number of sequences (snippets)
n_seq = frames_list.shape[-1]-1

# Number of time bins
n_t_sim = (2*int(flash_duration*stimulus_framerate)) * int(1/stimulus_framerate/bin_size)

# Process stimulus by batch
batch_size = 20 # must divide n_t_sim to not loose data
n_batch_sim = np.int(n_t_sim/batch_size)

# Loop over snippets
stimulus_fields_array = np.zeros((n_seq, n_t_sim, n_cells_synth))
for i_seq in tqdm(range(n_seq)):
    # Create stimulus frames
    stimulus_frames_sim = torch.ones((n_x, n_y, n_t_sim+n_bins_past_stim))
    stimulus_frames_sim[:,:,n_bins_past_stim+n_t_sim//2:] = torch.tensor(frames_list[:,:,i_seq+1][:,:,None])

    stimulus_fields = []
    for i_batch in range(n_batch_sim):
        # Prepare batch
        sample_indices = torch.concatenate([i_batch*batch_size + torch.arange(i_sample, i_sample + n_bins_past_stim) for i_sample in range(batch_size)])
        stimulus_frames_temp = torch.reshape(stimulus_frames_sim[:,:,sample_indices],
                                             (stimulus_frames_sim.shape[0], stimulus_frames_sim.shape[1], batch_size, n_bins_past_stim))

        # Project on raised cosine basis
        stimulus_frames_temp = stimulus_frames_temp.to(torch.float32).to(device)
        projected_stimulus = torch.matmul(stimulus_frames_temp, basis_stim.T).permute(2, 3, 0, 1)[:,None,:,:,:]
        
        # Loop over cells
        stimulus_fields_cells = []
        for i_cell in range(n_cells_synth):
        
            # Crop stimulus around the cell RF
            cell_RF_x = np.arange(population_positions[0,i_cell] - n_x_original//2,
                                  population_positions[0,i_cell] + n_x_original//2) + biasx_RF
            cell_RF_y = np.arange(population_positions[1,i_cell] - n_y_original//2,
                                  population_positions[1,i_cell] + n_y_original//2) + biasy_RF
                                  
            projected_stimulus_cell = projected_stimulus[:,:,:,cell_RF_x[:,None], cell_RF_y[None,:]]

            # Process through stimulus model
            pred = single_cell_model(projected_stimulus_cell)
            stimulus_fields_cells += [pred.detach().cpu().numpy().T]
            
        stimulus_fields_cells = np.concatenate(stimulus_fields_cells).T
        stimulus_fields += [stimulus_fields_cells]

    stimulus_fields = np.concatenate(stimulus_fields)
    stimulus_fields_array[i_seq,...] = stimulus_fields
    
# Permute axes
stimulus_fields_array = np.transpose(stimulus_fields_array, (2,1,0))

####################################################### SIMULATION ##########################################################
#############################################################################################################################

# Function and parameters to rebin spikes (saves memory before storage)
def rebin_spikes(array, new_bin_size):
    n_cells, n_time_bins, n_repeats = array.shape
    
    rebinned_time_bins = n_time_bins // new_bin_size
    
    reshaped_array = array[:,:rebinned_time_bins*new_bin_size,:].reshape(n_cells, rebinned_time_bins, new_bin_size, n_repeats)
    rebinned_array = np.sum(reshaped_array, axis=2)
    
    return rebinned_array
    
new_bin = 50
n_t_rebin = n_t_sim//new_bin

# Simulation parameters
n_rep_sim = 5000

# Break simulation in chunks if needed (in case of limited memory)
n_rep_chunks = 1
n_chunks = n_rep_sim//n_rep_chunks

# Combined fields
combined_filters = np.stack((history_filter, coupling_filter), axis=0)

# Nonlinearity
sigmoid = lambda x: 1/(1+np.exp(-x))

# Storage array
spikes_rebin = np.zeros((n_cells_synth, n_t_rebin, n_rep_sim, n_seq), dtype='uint8')

# Simulation loop
for i_chunk in tqdm(range(n_chunks)):
    # Initialize arrays
    rates_temp = np.zeros((n_cells_synth, n_rep_chunks, n_seq), dtype='float32')
    spikes_chunk = np.zeros((n_cells_synth, n_bins_past_int+n_t_sim, n_rep_chunks, n_seq), dtype='uint8')
    
    # Random table
    lookup_random = np.random.rand(n_cells_synth, n_t_sim, n_rep_chunks, n_seq)
    
    # Simulation loop
    for i_t in range(n_t_sim):
        
        # Interaction fields
        spikes_past_flat = spikes_chunk[:,i_t:n_bins_past_int+i_t,:,:].reshape(n_cells_synth, n_bins_past_int, n_rep_chunks * n_seq)
        combined_fields = np.tensordot(combined_filters, spikes_past_flat, axes=([1], [1]))
        history_fields, coupling_effect = combined_fields
        coupling_fields = connectivity_matrix @ coupling_effect
        int_fields = (history_fields + coupling_fields).reshape(n_cells_synth, n_rep_chunks, n_seq)

        # Firing rates
        rates_temp = sigmoid(int_fields + stimulus_fields_array[:,i_t,None,:])

        # Sort spikes
        spikes_temp = lookup_random[:,i_t,:,:] <= rates_temp
        
        # Store simulated spikes
        spikes_chunk[:,n_bins_past_int+i_t,:,:] = spikes_temp
        
    # Rebin spikes
    spikes_chunk_rebin = rebin_spikes(spikes_chunk[:,n_bins_past_int:,:,:].reshape(n_cells_synth, n_t_sim, n_rep_chunks*n_seq), new_bin)
    spikes_chunk_rebin = spikes_chunk_rebin.reshape(n_cells_synth, n_t_rebin, n_rep_chunks, n_seq)
    spikes_rebin[:,:,i_chunk*n_rep_chunks:(i_chunk+1)*n_rep_chunks,:] = spikes_chunk_rebin
    
# Keep spikes after stimulus onset
t_onset = 5 # Onset at 5 time bins
spikes_rebin = spikes_rebin[:,t_onset:,:,:]

# Save spikes
np.save(f'./{spikes_folder}/spikes_sim_{n_rep_sim}rep_{n_cells_synth}cells_{model_folder}.npy', spikes_rebin)
