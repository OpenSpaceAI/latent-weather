import os.path

import torch
import xarray as xr
from tqdm import trange

obs_path = "/path/to/dataset/earth_grid/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

data = xr.open_dataset(obs_path, engine="zarr")
data = data.transpose("time", "level", "latitude", "longitude")

var_list = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

save_path = "/path/to/dataset/earth_grid/tensor"

dtype = torch.float
device = "cpu"
kwargs = {"dtype": dtype, "device": device}

for i in trange(data.sizes["time"]):
    tensor = None
    for j in var_list:
        if len(data[j].dims) == 4:
            if tensor is None:
                tensor = torch.tensor(data[j][i].to_numpy(), **kwargs)
            else:
                t = torch.tensor(data[j][i].to_numpy(), **kwargs)
                tensor = torch.cat((tensor, t))
        elif len(data[j].dims) == 3:
            if tensor is None:
                tensor = torch.tensor(data[j][i].to_numpy(), **kwargs).unsqueeze(0)
            else:
                t = torch.tensor(data[j][i].to_numpy(), **kwargs).unsqueeze(0)
                tensor = torch.cat((tensor, t))
    torch.save(tensor, os.path.join(save_path, f"{i + 1:08d}.pt"))
