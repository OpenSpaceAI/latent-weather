import torch
from torch import nn


class RMSE(nn.Module):
    def __init__(self, type, device):
        super().__init__()

        if type == "mars_grid":
            lat = torch.load(
                "/path/to/dataset/mars_grid/ref/lat.pt",
                map_location=device,
                weights_only=True,
            )
        elif type == "earth_grid":
            lat = torch.load(
                "/path/to/dataset/earth_grid/ref/lat.pt",
                map_location=device,
                weights_only=True,
            )
        else:
            exit(1)

        weight = torch.cos(lat / 180 * torch.pi)
        weight = weight / torch.mean(weight)

        self.weight = weight.unsqueeze(1)

    def forward(self, predict, real):
        predict = predict.to(dtype=torch.float)
        real = real.to(dtype=torch.float)

        loss = (predict - real) ** 2

        loss = loss * self.weight
        loss = torch.mean(loss)
        loss = torch.sqrt(loss)

        return loss


class ACC(nn.Module):
    def __init__(self, type, device):
        super().__init__()

        if type == "mars_grid":
            lat = torch.load(
                "/path/to/dataset/mars_grid/ref/lat.pt",
                map_location=device,
                weights_only=True,
            )
        elif type == "earth_grid":
            lat = torch.load(
                "/path/to/dataset/earth_grid/ref/lat.pt",
                map_location=device,
                weights_only=True,
            )
        else:
            exit(1)

        weight = torch.cos(lat / 180 * torch.pi)
        weight = weight / torch.mean(weight)

        self.weight = weight.unsqueeze(1)

    def forward(self, predict, real, mean):
        predict = predict.to(dtype=torch.float)
        real = real.to(dtype=torch.float)

        predict = predict - mean
        real = real - mean

        frac_1 = torch.sum(predict * real * self.weight)
        frac_2 = torch.sqrt(
            torch.sum(predict**2 * self.weight) * torch.sum(real**2 * self.weight)
        )
        loss = frac_1 / frac_2

        return loss


def MAE(predict, real):
    predict = predict.to(dtype=torch.float)
    real = real.to(dtype=torch.float)

    loss = torch.nn.functional.l1_loss(predict, real)

    return loss


def MSE(predict, real):
    predict = predict.to(dtype=torch.float)
    real = real.to(dtype=torch.float)

    loss = torch.nn.functional.mse_loss(predict, real)

    return loss
