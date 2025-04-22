import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import math
import json  # Import json library
import argparse
import shutil
from tqdm import tqdm
import pandas as pd
import datetime
import gc
import zarr

from utils import *
from loss import *
from sampler import *

data_directory = '../data'
result_directory = './results'
model_directory = './models'

variable_names = ['z500', 't850', 't2m', 'u10', 'v10']
num_variables, num_static_fields = 5, 2
max_horizon = 240 # Maximum time horizon for the model. Used for scaling time embedding and making sure we don't go outside dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Run model with configuration from JSON file.')
parser.add_argument('config_path', type=str, help='Path to JSON configuration file.')
args = parser.parse_args()

def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config
config_path = args.config_path
config = load_config(config_path)

# Load config
name        =   config['name']
spacing     =   config['spacing']
t_direct =      config['t_direct']
t_max =         config['t_max']
batch_size =    config['batch_size']
t_min =         t_direct
t_iter =        config['t_iter']
n_ens =         config['n_ens']
model_path =    config['model']

print(name, flush=True)
print("[t_direct, t_iter, t_max]", [t_direct, t_iter, t_max],  flush=True)
print("n_ens:", n_ens,  flush=True)

# Copy config
result_path = Path(f'{result_directory}/{name}')
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)
shutil.copy(config_path, result_path / "config.json")

# Load normalization factors
with open(f'{data_directory}/norm_factors.json', 'r') as f:
    statistics = json.load(f)
mean_data = torch.tensor([stats["mean"] for (key, stats) in statistics.items() if key in variable_names])
std_data = torch.tensor([stats["std"] for (key, stats) in statistics.items() if key in variable_names])
norm_factors = np.stack([mean_data, std_data], axis=0)
mean_data = mean_data.to(device)
std_data = std_data.to(device)
def renormalize(x, mean_ar=mean_data, std_ar=std_data):
    x = x * std_ar[None, :, None, None] + mean_ar[None, :, None, None]
    return x

# Get the number of samples, training and validation samples
ti = pd.date_range(datetime.datetime(1979,1,1,0), datetime.datetime(2018,12,31,23), freq='1h')
n_samples, n_train, n_val = len(ti), sum(ti.year <= 2015), sum((ti.year >= 2016) & (ti.year <= 2017))

# Load the latitudes and longitudes
lat, lon = np.load(f'{data_directory}/latlon_1979-2018_5.625deg.npz').values()

# Load config of trained model
train_config_path = f'{model_directory}/{model_path}/config.json'
config = load_config(train_config_path)

# Constants and configurations loaded from JSON
filters     = config['filters']
max_trained_lead_time = config['t_max']
conditioning_times = config['conditioning_times']
delta_t = config['delta_t']
model_choice = config['model']

if t_iter > max_trained_lead_time:
    print(f"The iterative lead time {t_iter} is larger than the maximum trained lead time {max_trained_lead_time}")
if t_direct < delta_t:
    print(f"The direct lead time {t_direct} is smaller than the trained dt {delta_t}")

kwargs = {
            'dataset_path':     f'{data_directory}/z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy',
            'sample_counts':    (n_samples, n_train, n_val),
            'dimensions':       (num_variables, len(lat), len(lon)),
            'max_horizon':      max_horizon, # For scaling the time embedding
            'norm_factors':     norm_factors,
            'device':           device,
            'spacing':          spacing,
            'dtype':            'float32',
            'conditioning_times':    conditioning_times,
            'lead_time_range':  [t_min, t_max, t_direct],
            'static_data_path': f'{data_directory}/orog_lsm_1979-2018_5.625deg.npy',
            'random_lead_time': 0,
            }

input_times = (1 + len(conditioning_times))*num_variables + num_static_fields

if 'autoregressive' in model_choice:
    time_emb = 0
elif 'continuous' in model_choice:
    time_emb = 1
else:
    raise ValueError(f"Model choice {model_choice} not recognized.")

# Define the model and loss function
model = EDMPrecond(filters=filters, img_channels=input_times, out_channels=num_variables, img_resolution = 64, time_emb=time_emb, 
                    sigma_data=1, sigma_min=0.02, sigma_max=88)

model.load_state_dict(torch.load(f'{model_directory}/{model_path}/best_model.pth', map_location=device))
model.to(device)

print(f"Loaded model {model_path}, {model_choice}",  flush=True)
print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)

forecasting_times = t_min + t_direct * np.arange(0, 1 + (t_max-t_min)//t_direct)
dataset = ERA5Dataset(lead_time=forecasting_times, dataset_mode='test', **kwargs)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

sampler_fn = heun_sampler

print(f"Datset contains {len(dataset)} samples",  flush=True)
print(f"We do {len(loader)} batches",  flush=True)

model.eval()

# Initialize the dimensions based on the first batch
previous, current, time_labels = next(iter(loader))
n_times = time_labels.shape[1]
n_conditions = previous.shape[1]
dx = current.shape[2]
dy = current.shape[3]    

predictions = zarr.open(f'{result_path}/{name}.zarr', mode='w', shape=(len(dataset), n_ens, n_times, num_variables, dx, dy), 
                                chunks = (1, n_ens, n_times, num_variables, dx, dy),
                                dtype='float32', overwrite=True)

start_idx = 0  # Track index for where to write in the file

# Predict
for previous, current, time_labels in tqdm(loader):        
    n_samples = current.shape[0]

    with torch.no_grad():
        previous = previous.to(device)
        current = current.view(-1, num_variables, dx, dy).to(device)
        
        direct_time_labels = torch.tensor(np.array([x for x in time_labels[0] if x <= t_iter]), device=device)
        n_iter = time_labels.shape[1] // direct_time_labels.shape[0]
        n_direct = direct_time_labels.shape[0]

        class_labels = previous.repeat_interleave(n_direct * n_ens, dim=0) # Can not be changed if batchsz > 1

        static_fields = class_labels[:, -num_static_fields:]

        latent_shape = (n_samples * n_ens, num_variables, dx, dy)
        latents = torch.randn(latent_shape, device=device)

        direct_time_labels = direct_time_labels.repeat(n_ens * n_samples) # Can not be changed if n_direct > 1

        # Test
        predicted_combined = torch.zeros((n_samples, n_ens, n_times, num_variables, dx, dy), device=device)

        for i in tqdm(range(n_iter)):
            latents = torch.randn(latent_shape, device=device)
            latents = latents.repeat_interleave(n_direct, dim=0) # Can not be changed if batchsz > 1 or n_ens >1
            
            predicted = sampler_fn(model, latents, class_labels, direct_time_labels / max_horizon, 
                                    sigma_max=80, sigma_min=0.03, rho=7, num_steps=20, S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05)

            predicted_combined[:, :, i*n_direct:(i+1)*n_direct] = predicted.view(n_samples, n_ens, n_direct, num_variables, dx, dy)

            predicted = predicted.view(n_samples*n_ens, n_direct, num_variables, dx, dy)
            class_labels = class_labels.view(n_samples*n_ens, n_direct, n_conditions, dx, dy)[:, 0]

            if n_direct == 1:
                class_labels = torch.cat((predicted[:,-1], class_labels[:,:num_variables]), dim=1)#.repeat_interleave(n_direct, dim=0)
            else:
                class_labels = torch.cat((predicted[:,-1], predicted[:,-2]), dim=1).repeat_interleave(n_direct, dim=0) # Can not be changed if batchsz > 1
            
            if num_static_fields != 0:
                class_labels = torch.cat((class_labels, static_fields), dim=1)

        # Save predictions incrementally to zarr file
        predictions[start_idx:start_idx + n_samples, :, :, :, :, :] = renormalize(predicted_combined).view(n_samples, n_ens, n_times, num_variables, dx, dy).cpu().numpy()

        start_idx += n_samples

    gc.collect()
    torch.cuda.empty_cache()
    

# Calculate metrics
metrics = zarr.open_group('evaluation_metrics.zarr', mode='a')

calculate_WCRPS = calculate_AreaWeightedRMSE(lat, lon, device).CRPS
calculate_WScores = calculate_AreaWeightedRMSE(lat, lon, device).skill_and_spread
calculate_WMAE = calculate_AreaWeightedRMSE(lat, lon, device).mae

skill_list = []
spread_list = []
ssr_list = []
CRPS_list = []
dx_same_list = []
dx_different_list = []
dx_truth_list = []

i = 0
with torch.no_grad():
    for previous, current, time_labels in tqdm(loader):
        n_times = time_labels.shape[1]
        n_samples, _, dx, dy = current.shape

        truth = renormalize(current.to(device).view(n_samples, n_times, num_variables, dx, dy))
       
        forecast = predictions[i:i + truth.shape[0]]
        forecast = torch.tensor(forecast, device=device)
        i = i + truth.shape[0]
        
        # Add windspeed
        w_truth = (truth[:,:,3]**2 + truth[:,:,4]**2).sqrt().unsqueeze(2)
        truth = torch.cat((truth, w_truth), dim=2)
        w_forecast = (forecast[:,:,:,3]**2 + forecast[:,:,:,4]**2).sqrt().unsqueeze(3)
        forecast = torch.cat((forecast, w_forecast), dim=3)        
        
        # Calculate metrics
        skill, spread, ssr = calculate_WScores(forecast, truth)
        CRPS = calculate_WCRPS(forecast, truth)
        dx_same = calculate_WMAE(forecast[:, :, 1:, :], forecast[:, :, :-1, :])
        dx_different = calculate_WMAE(forecast[:, 1:, 1:, :], forecast[:, :-1, :-1, :])
        dx_truth = calculate_WMAE(truth[:, 1:, :].unsqueeze(1), truth[:, :-1, :].unsqueeze(1))


        # Append to list
        skill_list.append(skill)
        spread_list.append(spread)
        ssr_list.append(ssr)
        CRPS_list.append(CRPS)
        dx_same_list.append(dx_same)
        dx_different_list.append(dx_different)
        dx_truth_list.append(dx_truth)
        

skill = torch.tensor(np.array(skill_list)).mean(axis=0).cpu().numpy()
spread = torch.tensor(np.array(spread_list)).mean(axis=0).cpu().numpy()
ssr = torch.tensor(np.array(ssr_list)).mean(axis=0).cpu().numpy()
CRPS = torch.tensor(np.array(CRPS_list)).mean(axis=0).cpu().numpy()
dx_same = torch.tensor(np.array(dx_same_list)).mean(axis=0).cpu().numpy()
dx_different = torch.tensor(np.array(dx_different_list)).mean(axis=0).cpu().numpy()
dx_truth = torch.tensor(np.array(dx_truth_list)).mean(axis=0).cpu().numpy()

# Check if group for eval_name exists, else create it
if name not in metrics:
    metrics.create_group(name)

# Store the metrics in the corresponding group
group = metrics[name]
group.array('RMSE', skill, overwrite=True)
group.array('spread', spread, overwrite=True)
group.array('SSR', ssr, overwrite=True)
group.array('CRPS', CRPS, overwrite=True)
group.array('dx_same', dx_same, overwrite=True)
group.array('dx_different', dx_different, overwrite=True)
group.array('dx_truth', dx_truth, overwrite=True)
group.array('times', forecasting_times, overwrite=True)

print(f"Metrics saved under {name}")
