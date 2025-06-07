import math
import sys

import torch
from torch import nn

sys.path.append(".")

from earth_station_ae.model import Autoencoder


class Forecaster(nn.Module):
    def __init__(self, kwargs):
        super().__init__()

        self.autoencoder = Autoencoder(kwargs).requires_grad_(False)

        self.transformer = ForecasterTransformer(kwargs)

    def forward(self, input, query):
        input_shape = input.shape

        input = input.flatten(0, 1)
        input_latent = self.autoencoder.encoder(input)

        input_latent = input_latent.unflatten(0, (input_shape[0], input_shape[1]))

        predict_latent = self.transformer(input_latent, query)

        predict = self.autoencoder.decoder(predict_latent)

        return predict


class ForecasterTransformer(nn.Module):
    def __init__(self, kwargs):
        super().__init__()

        latent_size = 512

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_size,
            nhead=32,
            dim_feedforward=latent_size * 4,
            batch_first=True,
            activation="gelu",
            dropout=0.2,
            norm_first=False,
            **kwargs,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.pe = PositionalEncoding(latent_size, kwargs)

        self.q = 10000 ** (torch.arange(0, latent_size, 2, **kwargs) / latent_size)

    def forward(self, input, query):
        query = query / self.q
        query = torch.cat([torch.sin(query), torch.cos(query)], dim=1)
        query = query + input[:, -1]
        query = query.unsqueeze(1)

        input = self.pe(input)

        predict = self.decoder(query, input).squeeze(1)

        return predict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, kwargs):
        super().__init__()

        max_length = 5000

        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros(max_length, d_model, **kwargs)
        k = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1]].requires_grad_(False)
        x = self.dropout(x)

        return x
