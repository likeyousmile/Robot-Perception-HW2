import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary

from data import view, data_loader

EMBEDDING_DIM = 16
EPOCHS = 5


class CAEMedium(nn.Module):
    """3 Conv + 2 FC ~ 5K params"""

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(2, 2)
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=32 * 3 * 3, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=embedding_dim),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32 * 3 * 3),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(2, 2)
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=1,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            # nn.Sigmoid()
            nn.BatchNorm2d(num_features=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class CAESmall(nn.Module):
    """2 Conv + 1 FC ~ 2.5K params"""

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=4,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=4 * 7 * 7, out_features=embedding_dim),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=4 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(4, 7, 7)),

            nn.ConvTranspose2d(
                in_channels=4,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            # nn.Sigmoid()
            nn.BatchNorm2d(num_features=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def fit(network: nn.Module, train_data: DataLoader, loss_fn, opt_method):
    network.train()

    for image_batch, _label_batch in train_data:
        _encoded, decoded = network(image_batch)
        loss = loss_fn(decoded, image_batch)

        opt_method.zero_grad()
        loss.backward()
        opt_method.step()


def train(network: nn.Module, train_data: DataLoader, filename: str = None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-05)

    for epoch in range(EPOCHS):
        fit(network, train_data, criterion, optimizer)

    if filename:
        torch.save(net.state_dict(), filename)


if __name__ == "__main__":
    networks = [
        (CAEMedium(EMBEDDING_DIM), "./ConvAE-Medium-16.pt"),
        (CAEMedium(2), "./ConvAE-Medium-2.pt"),
        (CAESmall(2), "./ConvAE-Small-2.pt")
    ]

    for net, fn in networks:
        summary(net, input_size=(1, 28, 28))

        train(net, data_loader, fn)
        view.comparison(net)
