# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 512, 512]
        # Output size: [batch, 3, 512, 512]

        self.encoder = models.resnet50(True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 2048)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
			nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(48, 36, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(36, 24, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 6, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print(x.size())
        # print(encoded.size())

        encoded_reform = torch.reshape(encoded,  (-1, 128, 4, 4))
        # print(encoded_reform.size())
        decoded = self.decoder(encoded_reform)
        # print(decoded.size())
        return decoded, encoded