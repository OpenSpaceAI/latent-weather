import math

import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, kwargs):
        super().__init__()

        channel_num = 2
        latent_size = 512

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

        self.patch_1 = nn.Conv1d(
            in_channels, latent_size, kernel_size=4, stride=4, padding=2, **kwargs
        )
        self.patch_2 = nn.Conv1d(
            latent_size, latent_size, kernel_size=4, stride=4, padding=1, **kwargs
        )
        self.patch_3 = nn.Conv1d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=1, **kwargs
        )
        self.patch_4 = nn.Conv1d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=1, **kwargs
        )
        self.patch_5 = nn.Conv1d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=1, **kwargs
        )
        self.patch_6 = nn.Conv1d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=1, **kwargs
        )
        self.patch_7 = nn.Conv1d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=0, **kwargs
        )
        self.patch_8 = nn.Conv1d(
            latent_size, latent_size, kernel_size=2, stride=2, padding=0, **kwargs
        )

        self.decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_3 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_4 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_5 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_6 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_7 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_8 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.linear = nn.Sequential(
            nn.Linear(latent_size * 4, latent_size * 2, **kwargs),
            nn.LayerNorm(latent_size * 2, **kwargs),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 2, latent_size, **kwargs),
        )

    def forward(self, x):
        decoder_1_raw = self.patch_1(x)
        decoder_1 = decoder_1_raw.transpose(1, 2)

        patch_2 = self.patch_2(decoder_1_raw).transpose(1, 2)
        decoder_2 = self.decoder_2(patch_2, decoder_1)
        decoder_2_raw = decoder_2.transpose(1, 2)

        patch_3 = self.patch_3(decoder_2_raw).transpose(1, 2)
        decoder_3 = self.decoder_3(patch_3, decoder_2)
        decoder_3_raw = decoder_3.transpose(1, 2)

        patch_4 = self.patch_4(decoder_3_raw).transpose(1, 2)
        decoder_4 = self.decoder_4(patch_4, decoder_3)
        decoder_4_raw = decoder_4.transpose(1, 2)

        patch_5 = self.patch_5(decoder_4_raw).transpose(1, 2)
        decoder_5 = self.decoder_5(patch_5, decoder_4)
        decoder_5_raw = decoder_5.transpose(1, 2)

        patch_6 = self.patch_6(decoder_5_raw).transpose(1, 2)
        decoder_6 = self.decoder_6(patch_6, decoder_5)
        decoder_6_raw = decoder_6.transpose(1, 2)

        patch_7 = self.patch_7(decoder_6_raw).transpose(1, 2)
        decoder_7 = self.decoder_7(patch_7, decoder_6)
        decoder_7_raw = decoder_7.transpose(1, 2)

        patch_8 = self.patch_8(decoder_7_raw).transpose(1, 2)
        decoder_8 = self.decoder_8(patch_8, decoder_7).flatten(1)

        x = self.linear(decoder_8)

        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_size, kwargs):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2, **kwargs),
            nn.LayerNorm(latent_size * 2, **kwargs),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 2, latent_size * 4, **kwargs),
        )

        self.deconv_8 = nn.ConvTranspose1d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            **kwargs,
        )
        self.deconv_7 = nn.ConvTranspose1d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            **kwargs,
        )
        self.deconv_6 = nn.ConvTranspose1d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=1,
            output_padding=1,
            **kwargs,
        )
        self.deconv_5 = nn.ConvTranspose1d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=1,
            output_padding=1,
            **kwargs,
        )
        self.deconv_4 = nn.ConvTranspose1d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=1,
            output_padding=1,
            **kwargs,
        )
        self.deconv_3 = nn.ConvTranspose1d(
            latent_size,
            latent_size,
            kernel_size=2,
            stride=2,
            padding=1,
            output_padding=1,
            **kwargs,
        )
        self.deconv_2 = nn.ConvTranspose1d(
            latent_size,
            latent_size,
            kernel_size=4,
            stride=4,
            padding=1,
            output_padding=1,
            **kwargs,
        )
        self.deconv_1 = nn.ConvTranspose1d(
            latent_size,
            out_channels,
            kernel_size=4,
            stride=4,
            padding=2,
            output_padding=2,
            **kwargs,
        )

        self.norm = nn.LayerNorm(latent_size, **kwargs)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear(x).unflatten(1, (-1, 4))

        x = self.act(self.norm(self.deconv_8(x).transpose(1, 2))).transpose(1, 2)
        x = self.act(self.norm(self.deconv_7(x).transpose(1, 2))).transpose(1, 2)
        x = self.act(self.norm(self.deconv_6(x).transpose(1, 2))).transpose(1, 2)
        x = self.act(self.norm(self.deconv_5(x).transpose(1, 2))).transpose(1, 2)
        x = self.act(self.norm(self.deconv_4(x).transpose(1, 2))).transpose(1, 2)
        x = self.act(self.norm(self.deconv_3(x).transpose(1, 2))).transpose(1, 2)
        x = self.act(self.norm(self.deconv_2(x).transpose(1, 2))).transpose(1, 2)
        x = self.deconv_1(x)

        return x
