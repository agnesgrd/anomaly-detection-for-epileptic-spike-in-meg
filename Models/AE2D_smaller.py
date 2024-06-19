import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x = nn.functional.pad(x, (2, 2, 3, 3))
        encoded = self.encoder(x)
        return encoded


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        # decoded = torch.narrow(decoded, -1, 1, 30)
        decoded = torch.narrow(decoded, -2, 4, 274)
        decoded = torch.narrow(decoded, -1, 2, 300)
        return decoded


class Autoencoder(nn.Module):
    def __init__(self, first_direction, sfreq, window_size):
        super(Autoencoder, self).__init__()

        if first_direction == "channel":
            input_shape = (1, 274, int(sfreq * window_size))
        elif first_direction == "time":
            input_shape = (1, int(sfreq * window_size), 274)

        self.encoder = Encoder()
        self.decoder = Decoder()  # Get output channels of the last conv layer

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
