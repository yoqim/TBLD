import math
import torch.nn as nn
from torch.nn import init

# import pdb;pdb.set_trace()

class BrownFcModel(nn.Module):
    def __init__(self, G_bit, F_bit):
        super(BrownFcModel, self).__init__()
        self.G_bit = G_bit
        self.F_bit = F_bit
        self.G = nn.Sequential(
            nn.Linear(4096, self.G_bit),
        )
        self.F = nn.Sequential(
            nn.Linear(self.G_bit, self.F_bit),
        )

        nn.init.kaiming_normal_(self.G[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.F[0].weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        G_out = self.G(x)
        F_out = self.F(G_out)

        G_out = G_out.t()
        F_out = F_out.t()
        return G_out, F_out

class Discriminator(nn.Module):
    def __init__(self, hash_bit):
        super(Discriminator, self).__init__()
        h_dim = [512, 128]
        self.discriminator = nn.Sequential(
            nn.Linear(hash_bit, h_dim[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h_dim[0], h_dim[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h_dim[1], 1),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x
