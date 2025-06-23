import os.path

import natsort
import torch
import xarray as xr

data_path = "/path/to/dataset/mars_grid/raw"
data_list = os.listdir(data_path)
data_list = natsort.natsorted(data_list)

save_path = "/path/to/dataset/mars_grid/tensor"

dtype = torch.float
device = "cpu"
kwargs = {"dtype": dtype, "device": device}

cnt = 0

for i in data_list:
    data = xr.open_dataset(os.path.join(data_path, i))

    ps = torch.tensor(data["ps"].to_numpy(), **kwargs).unsqueeze(1)
    tsurf = torch.tensor(data["tsurf"].to_numpy(), **kwargs).unsqueeze(1)
    co2ice = torch.tensor(data["co2ice"].to_numpy(), **kwargs).unsqueeze(1)
    dustcol = torch.tensor(data["dustcol"].to_numpy(), **kwargs).unsqueeze(1)

    temp = torch.tensor(data["temp"].to_numpy(), **kwargs)
    u = torch.tensor(data["u"].to_numpy(), **kwargs)
    v = torch.tensor(data["v"].to_numpy(), **kwargs)

    tensor = torch.cat((ps, tsurf, co2ice, dustcol, temp, u, v), dim=1)

    for j in range(tensor.shape[0]):
        cnt += 1
        torch.save(tensor[j], os.path.join(save_path, f"{cnt:08d}.pt"))
