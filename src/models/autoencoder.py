import torch
import torch.nn as nn
from typing import List

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.config import ENCODER_CONV_CHANNELS, DECODER_CONV_CHANNELS

class AutoEncoder(nn.Module):
    """Autoencoder model."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        encoder_conv_channels: List[int] = ENCODER_CONV_CHANNELS,
        decoder_conv_channels: List[int] = DECODER_CONV_CHANNELS,
        activation: nn.Module = nn.ReLU(inplace=True),
        dropout_rate: float = 0.0,
        with_pooling: bool = True,
    ) -> None:
        """
        Initialize the autoencoder model.

        Args:
            input_size (int): Input size.
            output_size (int): Output size.
            encoder_conv_channels (List[int], optional): List of encoder convolutional channels. Defaults to ENCODER_CONV_CHANNELS.
            decoder_conv_channels (List[int], optional): List of decoder convolutional channels. Defaults to DECODER_CONV_CHANNELS.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            with_pooling (bool, optional): Whether to use pooling. Defaults to True.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.encoder_conv_channels = encoder_conv_channels
        self.decoder_conv_channels = decoder_conv_channels
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.with_pooling = with_pooling

        self.encoder = Encoder(
            input_size=input_size,
            conv_channels=encoder_conv_channels,
            activation=activation,
            dropout_rate=dropout_rate,
            with_pooling=with_pooling,
        )

        self.decoder = Decoder(
            output_size=output_size,
            conv_channels=decoder_conv_channels,
            activation=activation,
            dropout_rate=dropout_rate,
            with_pooling=with_pooling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)

        return x
