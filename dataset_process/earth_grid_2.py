import os

import torch
from tqdm import tqdm

data_dir = "/path/to/dataset/earth_grid/tensor"

data_list = [f"{i:08d}.pt" for i in range(80357, 87661)]

device = "cpu"

mean = torch.load(
    "/path/to/dataset/earth_grid/ref/mean.pt", map_location=device, weights_only=True
)
std = torch.load(
    "/path/to/dataset/earth_grid/ref/std.pt", map_location=device, weights_only=True
)

for data in tqdm(data_list, bar_format="{l_bar}{bar}{r_bar} ETA: {eta:%y/%m/%d %H:%M}"):
    t = torch.load(os.path.join(data_dir, data), map_location=device, weights_only=True)
    t = (t - mean) / std
    torch.save(t, os.path.join(data_dir, data))
