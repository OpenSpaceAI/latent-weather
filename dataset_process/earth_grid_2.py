import os

import torch
from tqdm import tqdm

data_dir = "/path/to/dataset/earth_grid/tensor"

data_list = os.listdir(data_dir)

device = "cpu"

mean = torch.load(
    "/path/to/dataset/earth_grid/ref/mean.pt", map_location=device, weights_only=True
)
std = torch.load(
    "/path/to/dataset/earth_grid/ref/std.pt", map_location=device, weights_only=True
)

for data in tqdm(data_list):
    t = torch.load(os.path.join(data_dir, data), map_location=device, weights_only=True)
    t = (t - mean) / std
    torch.save(t, os.path.join(data_dir, data))
