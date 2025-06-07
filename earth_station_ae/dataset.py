import os

import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, type):
        self.device = "cpu"

        data_dir = "/path/to/dataset/earth_station/tensor"

        if type == "train":
            data_name = "train.pt"
        elif type == "val":
            data_name = "val.pt"
        else:
            exit(1)

        self.data = torch.load(
            os.path.join(data_dir, data_name),
            map_location=self.device,
            weights_only=True,
        )

        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input_raw = self.data[index]

        data = {"input_raw": input_raw}

        return data
