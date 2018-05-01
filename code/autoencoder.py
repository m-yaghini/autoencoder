from torch import nn


class Autoencoder(nn.Module):
    '''
    Main class of autoencoder. The constructer requires the number of input features.
    The model architecture is a two-hidden layer, symmetrical autoencoder with batch normalization and ReLU activations.
    '''

    def __init__(self, numof_features):
        super(Autoencoder, self).__init__()
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
