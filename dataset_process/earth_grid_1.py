import datetime
import os.path

import torch
import xarray as xr
from torch import distributed, multiprocessing
from tqdm import trange


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"

    distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(hours=1),
    )


def cleanup():
    distributed.destroy_process_group()


def main(rank, world_size):
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

    per = data.sizes["time"] // world_size + 1
    start = rank * per
    end = min(data.sizes["time"], (rank + 1) * per)

    if rank == 0:
        range_bar = trange(
            start, end, bar_format="{l_bar}{bar}{r_bar} ETA: {eta:%y/%m/%d %H:%M}"
        )
    else:
        range_bar = range(start, end)

    print(f"Rank {rank}: range {start}-{end}")

    for i in range_bar:
        if os.path.exists(
            os.path.join(save_path, f"{i + 1:08d}.pt")
        ) and os.path.exists(os.path.join(save_path, f"{i + 2:08d}.pt")):
            continue
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
        if torch.isnan(tensor).any():
            print(f"Rank {rank}: {i + 1:08d}.pt nan")
            exit(1)
        if torch.isinf(tensor).any():
            print(f"Rank {rank}: {i + 1:08d}.pt inf")
            exit(1)
        torch.save(tensor, os.path.join(save_path, f"{i + 1:08d}.pt"))


if __name__ == "__main__":
    world_size = 4

    multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
