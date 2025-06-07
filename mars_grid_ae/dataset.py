import os

import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, type):
        self.device = "cpu"

        self.data_dir = "/path/to/dataset/mars_grid/tensor"

        self.data_list = os.listdir(self.data_dir)
        self.data_list.sort()

        train_num = int(len(self.data_list) * 0.8)
        if type == "train":
            self.data_list = self.data_list[:train_num]
        elif type == "val":
            self.data_list = self.data_list[train_num:]
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
