from torch import nn


class autoencoder(nn.Module):
    def __init__(self, numof_features):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(numof_features, 200, bias=False),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Linear(200, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 50, bias=False),
            nn.BatchNorm1d(50),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(50, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 200, bias=False),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Linear(200, numof_features, bias=False),
            nn.BatchNorm1d(numof_features))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x