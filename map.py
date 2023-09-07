import torch
import numpy as np

from torch import nn


from args import Args
from grid import Grid

class Map(nn.Module):
    def __init__(self, args:Args) -> None:
        self.args = args
        
        # create learnable girds
        self.grids = []
        for _ in range(self.args.L):
            self.grids.append(Grid(self.args))

        # density neural network
        self.density_lin1 = nn.Linear(self.args.L*self.args.F, 64)  # L*F -> 64
        self.density_lin2 = nn.Linear(64, 16)

        # colour neural network
        self.colour_lin1 = nn.Linear(self.args.L*self.args.F + 2*self.args.D*self.args.f, 64)
        self.colour_lin2 = nn.Linear(64, 64)
        self.colour_lin3 = nn.Linear(64, 3)

        # activation fcts.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, X, D):
        """
        Forward pass of the map
        Args:
            X: batch of points; torch.tensor (I*R*M, D)
            D: batch of directions, normalized vectors indicating direction of view; torch.tensor (I*R, D)
        Returns:
            density: batch of densities from point X; torch.tensor (I*R*M, 1)
            colour: batch of colours from point X and with viewing direction D; torch.tensor (I*R*M, 3)
        """
        # concatenate encoding from every layer
        X_encoded = torch.empty(X.shape[0], 0)
        for grid in self.grids:
            X_encoded = torch.cat((X_encoded, grid.forward(X)), dim=1)

        # density prediction, vector of length 16 where the first element is the density
        density = self.relu(self.density_lin1(X_encoded))
        density = self.sigmoid(self.density_lin2(density))

        # encode direction
        D_encoded = self._encodeDirection(D)
        D_encoded = torch.cat((density, D_encoded), dim=1)

        # colour prediction
        colour = self.relu(self.colour_lin1(D_encoded))
        colour = self.relu(self.colour_lin2(colour))
        colour = self.sigmoid(self.colour_lin3(colour))

        return density[:,0], colour      

    def _encodeDirection(self, D):
        """
        Encode direction vector D
        Args:
            D: batch of directions, normalized vectors indicating direction of view; torch.tensor (I*R, D)
        Returns:
            D_encoded: torch.tensor (I*R*M, f*D)
        """
        # encode direction
        D_encoded = torch.empty(D.shape[0], 0)
        for i in range(self.args.f):
            D_encoded = torch.cat((D_encoded, torch.sin(2**i * torch.pi * D)), dim=1)
            D_encoded = torch.cat((D_encoded, torch.cos(2**i * torch.pi * D)), dim=1)

        # extand direction to match positions dimension
        D_encoded = D_encoded.repeat(self.args.M, 1) # (I*R*M, D)

        return D_encoded

    






        


        