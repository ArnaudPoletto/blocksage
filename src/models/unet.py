import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        # Define the encoder layers
        self.encoder_conv1 = nn.Conv3d(1006, 256, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.encoder_maxpool = nn.MaxPool3d(2)

        # Define the decoder layers
        self.decoder_conv1 = nn.ConvTranspose3d(1024, 256, kernel_size=2, stride=2)
        self.decoder_conv2 = nn.ConvTranspose3d(768, 256, kernel_size=2, stride=2)
        self.decoder_conv3 = nn.Conv3d(512, 1006, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.encoder_conv1(x))
        x2 = self.encoder_maxpool(x1)
        x2 = self.relu(self.encoder_conv2(x2))
        x3 = self.encoder_maxpool(x2)
        x3 = self.relu(self.encoder_conv3(x3))

        # Decoder
        x = self.relu(self.decoder_conv1(x3))
        x = torch.cat([x, x2], dim=1)  # Concatenate skip connection
        x = self.relu(self.decoder_conv2(x))
        x = torch.cat([x, x1], dim=1)  # Concatenate skip connection
        x = self.relu(self.decoder_conv3(x))

        # Only take the center section of the prediction
        # cluster_size = x.shape[2] // SECTION_SIZE
        # center_section_start = SECTION_SIZE * (cluster_size // 2)
        # center_section_end = center_section_start + SECTION_SIZE
        # x = x[
        #     :, # batch
        #     :, # classes
        #     center_section_start:center_section_end, 
        #     center_section_start:center_section_end, 
        #     center_section_start:center_section_end
        # ]

        return x