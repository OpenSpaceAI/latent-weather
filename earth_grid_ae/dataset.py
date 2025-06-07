import os

import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, type):
        self.device = "cpu"

        self.data_dir = "/path/to/dataset/earth_grid/tensor"

        if type == "train":
            self.data_list = [f"{i:08d}.pt" for i in range(80357, 86201)]
        elif type == "val":
            self.data_list = [f"{i:08d}.pt" for i in range(86201, 87661)]
        else:
            exit(1)

        self.len = len(self.data_list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input_name = self.data_list[index]

        input_raw = torch.load(
            os.path.join(self.data_dir, input_name),
            map_location=self.device,
            weights_only=True,
        )

        data = {"input_raw": input_raw}

        return data
