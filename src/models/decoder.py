import torch
import torch.nn as nn
from typing import List

from src.config import DECODER_CONV_CHANNELS

        
class Decoder(nn.Module):
    """Decoder for the autoencoder model."""

    def __init__(
        self,
        output_size: int,
        conv_channels: List[int] = DECODER_CONV_CHANNELS,
        activation: nn.Module = nn.ReLU(inplace=True),
        dropout_rate: float = 0.0,
        with_pooling: bool = True,
    ) -> None:
        """
        Initialize the encoder.

        Args:
            output (int): Output size.
            conv_channels (List[int], optional): List of convolutional channels. Defaults to CONV_CHANNELS.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            with_pooling (bool, optional): Whether to use pooling. Defaults to True.
        """
        super().__init__()

        self.output_size = output_size
        self.conv_channels = conv_channels
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.with_pooling = with_pooling

        self.decoder = nn.Sequential()

        conv_channels = conv_channels + [output_size]
        for i, (in_conv_channel, out_cov_channel) in enumerate(zip(conv_channels[:-1], conv_channels[1:])):
            if with_pooling:
                self.decoder.add_module(
                    f"conv_transpose{i}",
                    nn.ConvTranspose3d(
                        in_channels = in_conv_channel,
                        out_channels = out_cov_channel,
                        kernel_size = 2,
                        stride = 2,
                    ),
                )
            else:
                self.decoder.add_module(
                    f"conv{i}",
                    nn.Conv3d(
                        in_channels = in_conv_channel,
                        out_channels = out_cov_channel,
                        kernel_size = 3,
                        padding = 1,
                    ),
                )
            self.decoder.add_module(f"batchnorm{i}", nn.BatchNorm3d(out_cov_channel))
            self.decoder.add_module(f"dropout{i}", nn.Dropout3d(p=dropout_rate))
            self.decoder.add_module(f"activation{i}", activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.decoder(x)
