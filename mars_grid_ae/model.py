import math

import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, latent_size, kwargs):
        super().__init__()

        channel_num = 109

        self.encoder = Encoder(channel_num, latent_size, kwargs)
        self.decoder = Decoder(channel_num, latent_size, kwargs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_size, kwargs):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_size,
            nhead=32,
            dim_feedforward=latent_size * 4,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
            norm_first=False,
            **kwargs,
        )

        self.patch_1 = nn.Conv2d(
            in_channels, latent_size, kernel_size=2, stride=2, padding=(0, 0), **kwargs
        )
        self.patch_2 = nn.Conv2d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=(0, 0), **kwargs
        )
        self.patch_3 = nn.Conv2d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=(1, 0), **kwargs
        )
        self.patch_4 = nn.Conv2d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=(1, 1), **kwargs
        )
        self.patch_5 = nn.Conv2d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=(1, 1), **kwargs
        )

        self.decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_3 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_4 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_5 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.linear = nn.Sequential(
            nn.Linear(latent_size * 6, latent_size * 4, **kwargs),
            nn.LayerNorm(latent_size * 4, **kwargs),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 4, latent_size, **kwargs),
        )

    def forward(self, x):
        decoder_1_raw = self.patch_1(x)
        decoder_1 = decoder_1_raw.flatten(2).transpose(1, 2)

        patch_2 = self.patch_2(decoder_1_raw).flatten(2).transpose(1, 2)
        decoder_2 = self.decoder_2(patch_2, decoder_1)
        decoder_2_raw = decoder_2.transpose(1, 2).unflatten(2, (9, 18))

        patch_3 = self.patch_3(decoder_2_raw).flatten(2).transpose(1, 2)
        decoder_3 = self.decoder_3(patch_3, decoder_2)
        decoder_3_raw = decoder_3.transpose(1, 2).unflatten(2, (5, 9))

        patch_4 = self.patch_4(decoder_3_raw).flatten(2).transpose(1, 2)
        decoder_4 = self.decoder_4(patch_4, decoder_3)
        decoder_4_raw = decoder_4.transpose(1, 2).unflatten(2, (3, 5))

        patch_5 = self.patch_5(decoder_4_raw).flatten(2).transpose(1, 2)
        decoder_5 = self.decoder_5(patch_5, decoder_4).flatten(1)

        x = self.linear(decoder_5)

        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_size, kwargs):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4, **kwargs),
            nn.LayerNorm(latent_size * 4, **kwargs),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 4, latent_size * 6, **kwargs),
        )

        self.deconv_5 = nn.ConvTranspose2d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=(1, 1),
            output_padding=(1, 1),
            **kwargs,
        )
        self.deconv_4 = nn.ConvTranspose2d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=(1, 1),
            output_padding=(1, 1),
            **kwargs,
        )
        self.deconv_3 = nn.ConvTranspose2d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=(1, 0),
            output_padding=(1, 0),
            **kwargs,
        )
        self.deconv_2 = nn.ConvTranspose2d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=(0, 0),
            output_padding=(0, 0),
            **kwargs,
        )
        self.deconv_1 = nn.ConvTranspose2d(
            latent_size,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=(0, 0),
            output_padding=(0, 0),
            **kwargs,
        )

        self.norm = nn.LayerNorm(latent_size, **kwargs)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear(x).unflatten(1, (-1, 2, 3))

        x = (
            self.act(self.norm(self.deconv_5(x).flatten(2).transpose(1, 2)))
            .transpose(1, 2)
            .unflatten(2, (3, 5))
        )
        x = (
            self.act(self.norm(self.deconv_4(x).flatten(2).transpose(1, 2)))
            .transpose(1, 2)
            .unflatten(2, (5, 9))
        )
        x = (
            self.act(self.norm(self.deconv_3(x).flatten(2).transpose(1, 2)))
            .transpose(1, 2)
            .unflatten(2, (9, 18))
        )
        x = (
            self.act(self.norm(self.deconv_2(x).flatten(2).transpose(1, 2)))
            .transpose(1, 2)
            .unflatten(2, (18, 36))
        )
        x = self.deconv_1(x)

        return x
