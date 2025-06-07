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

        self.distance = 24

        self.past_num = 7
        self.future_num = 28
        self.future_range = list(
            range(self.distance, self.future_num * self.distance + 1, self.distance)
        )

        self.len = (
            self.data.shape[0] - (self.past_num + self.future_num - 1) * self.distance
        ) * self.future_num

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        query = self.future_range[index % self.future_num]

        input_start = index // self.future_num
        input_end = input_start + self.past_num * self.distance

        target_index = input_end - self.distance + query

        input_raw = self.data[input_start : input_end : self.distance]
        target_raw = self.data[target_index]

        query = torch.tensor(query, device=self.device, dtype=torch.float).unsqueeze(0)

        data = {"input_raw": input_raw, "query": query, "target_raw": target_raw}

        return data
