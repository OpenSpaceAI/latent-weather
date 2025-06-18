import os

import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, latent_name, type):
        self.latent = torch.load(
            f"result/{latent_name}/latent.pt", map_location="cpu", weights_only=True
        )

        self.data_dir = "/path/to/dataset/earth_grid/tensor"

        if type == "train":
            self.data_list = [f"{i:08d}.pt" for i in range(80357, 86201)]
            self.latent = self.latent[: 86201 - 80357]
        elif type == "val":
            self.data_list = [f"{i:08d}.pt" for i in range(86201, 87661)]
            self.latent = self.latent[86201 - 80357 :]
        else:
            exit(1)

        self.distance = 1

        self.past_num = 28
        self.future_num = 112
        self.future_range = list(
            range(self.distance, self.future_num * self.distance + 1, self.distance)
        )

        self.len = (
            len(self.data_list) - (self.past_num + self.future_num - 1) * self.distance
        ) * self.future_num

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        query = self.future_range[index % self.future_num]

        input_start = index // self.future_num
        input_end = input_start + self.past_num * self.distance

        target_index = input_end - self.distance + query

        target_name = self.data_list[target_index]

        input_raw = self.latent[input_start : input_end : self.distance]
        target_raw = torch.load(
            os.path.join(self.data_dir, target_name),
            map_location="cpu",
            weights_only=True,
        )

        query = torch.tensor(query, device="cpu", dtype=torch.float).unsqueeze(0)

        data = {"input_raw": input_raw, "query": query, "target_raw": target_raw}

        return data
