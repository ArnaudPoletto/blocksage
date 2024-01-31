import torch
import torch.nn as nn
from typing import List

from src.config import ENCODER_CONV_CHANNELS


class Encoder(nn.Module):
    """Encoder for the autoencoder model."""

    def __init__(
        self,
        input_size: int,
        conv_channels: List[int] = ENCODER_CONV_CHANNELS,
        activation: nn.Module = nn.ReLU(inplace=True),
        dropout_rate: float = 0.0,
        with_pooling: bool = True,
    ) -> None:
        """
        Initialize the encoder.

        Args:
            input_size (int): Input size.
            conv_channels (List[int], optional): List of convolutional channels. Defaults to CONV_CHANNELS.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            with_pooling (bool, optional): Whether to use pooling. Defaults to True.
        """
        super().__init__()

        self.input_size = input_size
        self.conv_channels = conv_channels
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.with_pooling = with_pooling

        self.encoder = nn.Sequential()

        conv_channels = [input_size] + conv_channels
        for i, (in_conv_channel, out_cov_channel) in enumerate(zip(conv_channels[:-1], conv_channels[1:])):
            self.encoder.add_module(
                f"conv{i}",
                nn.Conv3d(
                    in_channels = in_conv_channel,
                    out_channels = out_cov_channel,
                    kernel_size = 3,
                    padding = 1,
                ),
            )
            self.encoder.add_module(f"batchnorm{i}", nn.BatchNorm3d(out_cov_channel))
            self.encoder.add_module(f"dropout{i}", nn.Dropout3d(p=dropout_rate))
            self.encoder.add_module(f"activation{i}", activation)
            if with_pooling:
                self.encoder.add_module(f"pooling{i}", nn.MaxPool3d(kernel_size=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.encoder(x)
