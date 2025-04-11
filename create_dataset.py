import xarray as xr
import numpy as np
import json
from tqdm import tqdm

file_directory = "ERA5_DATA_LOCATION"
save_directory = "./data"

field_names = {
    "orography": "orog",
    "lsm": "lsm",
}

var_names = {
    "geopotential_500": ("z500", "z"),
    "temperature_850": ("t850", "t"),
    "2m_temperature": ("t2m", "t2m"),
    "10m_u_component_of_wind": ("u10", "u10"),
    "10m_v_component_of_wind": ("v10", "v10"),
}

chunk_size = 1000

# Constants
var_name = "constants"
file_pattern = f"{file_directory}/{var_name}/{var_name}*.nc"
df = xr.open_mfdataset(file_pattern, combine='by_coords')

lat = df['lat'].values
lon = df['lon'].values
np.savez(f'{save_directory}/latlon_1979-2018_5.625deg.npz', lat=lat, lon=lon)

## Static fields
static_fields = []
save_name = ''
for field_name, var_name in field_names.items():
    data_array = df[field_name].values
    static_fields.append(data_array)
    save_name += var_name + '_'

np.save(f'{save_directory}/{save_name}1979-2018_5.625deg.npy', np.stack(static_fields, axis=0))

# Variables
file_prefix, names = list(var_names.items())[0]
var_name = names[0]
short_name = names[1]
file_pattern = f"{file_directory}/{file_prefix}/{file_prefix}*.nc"
ds = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={'lat': 32, 'lon': 64})
combined_shape = (ds[short_name].shape[0], len(var_names), ds[short_name].shape[1], ds[short_name].shape[2])
print("Shape:", combined_shape)

save_name = '_'.join([var_name[0] for var_name in var_names.values()])
memmap_file_path = f'{save_directory}/{save_name}_1979-2018_5.625deg.npy'
memmap_array = np.memmap(memmap_file_path, dtype='float32', mode='w+', shape=combined_shape)

statistics = {}
mean_value = 0
std_value = 0

i = 0
for file_prefix, names in var_names.items():
    var_name = names[0]
    short_name = names[1]

    file_pattern = f"{file_directory}/{file_prefix}/{file_prefix}*.nc"
    print(f"Opening: {file_pattern}")
    
    # Open the dataset with dask for efficient memory handling
    ds = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={'lat': 32, 'lon': 64})
    array = ds[short_name]
    
    # Loop through the chunks of the array and write each chunk to the memmap file
    # We will iterate through the time dimension (0th axis) chunk by chunk
    for j in tqdm(range(0, array.shape[0], chunk_size)):
        end_idx = min(j + chunk_size, array.shape[0])  # Ensure we don't go out of bounds
        
        # Convert the chunk to a NumPy array and assign it to the corresponding memmap slice
        chunk = array[j:end_idx, :, :].compute()  # Compute the chunk lazily
        memmap_array[j:end_idx, i, :, :] = chunk

        # Calculate the mean and std of the chunk
        mean_value += np.sum(chunk).values
        std_value += np.sum(chunk ** 2).values
    
    # Calculate the mean and std of the variable
    num_elements = array.size
    mean_value /= num_elements
    std_value = np.sqrt(std_value / num_elements - mean_value ** 2)
    statistics[var_name] = {"mean": mean_value, "std": std_value}

    i += 1

print(f"Combined data saved as memory-mapped file: {memmap_file_path}")

# Save the statistics to a JSON file
json_file = f'{save_directory}/norm_factors.json'
with open(json_file, 'w') as f:
    json.dump(statistics, f, indent=4)

print(f"Normalization factors saved to {json_file}")

# Print out the mean and std for each variable
for var_name, stats in statistics.items():
    print(f"{var_name}: Mean = {stats['mean']}, Std = {stats['std']}")

