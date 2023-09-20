import torch
import numpy as np

from torch import nn
from activation import trunc_exp


from args import Args
from grid import GridHash, GridTable

class Map(nn.Module):
    def __init__(self, args:Args) -> None:
        # init. nn.Module
        super(Map, self).__init__()

        self.args = args
        
        # create learnable girds
        self.grids = []
        for i in range(self.args.L):
            # self.grids.append(GridHash(args=self.args, layer=i))
            self.grids.append(GridTable(args=self.args, layer=0))

        # Register parameters of the Grid instance as parameters of the Map instance
        for i, grid in enumerate(self.grids):
            for name, param in grid.named_parameters():
                self.register_parameter(name+str(i), param)

        # density neural network
        self.density_lin1 = nn.Linear(self.args.L*self.args.F, 64)  # L*F -> 64
        self.density_lin2 = nn.Linear(64, 16)

        # colour neural network
        self.colour_lin1 = nn.Linear(16 + 2*self.args.f*self.args.D, 64)
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
            density: batch of densities from point X; torch.tensor (I*R*M,)
            colour: batch of colours from point X and with viewing direction D; torch.tensor (I*R*M, 3)
        """
        # estimate density
        density, geo_features = self.forwardDensity(X) # (I*R*M,), (I*R*M, 15)

        # estimate colour
        colour = self.forwardColour(geo_features, D) # (I*R*M, 3)

        return density[:,0], colour
    
    def forwardDensity(self, X):
        """
        Forward pass to get the density
        Args:
            X: batch of points; torch.tensor (I*R*M, D)
        Returns:
            density: batch of densities from point X; torch.tensor (I*R*M,)
            geo_features: batch of geo_features from point X; torch.tensor (I*R*M, 15)
        """
        # concatenate encoding from every layer
        X_encoded = self._encodePosition(X) # (I*R*M, L*F)

        # density prediction, vector of length 16 where the first element is the density
        X_encoded = self.relu(self.density_lin1(X_encoded))
        X_encoded = self.sigmoid(self.density_lin2(X_encoded))

        density = trunc_exp(X_encoded[:,0]) # (I*R*M,)
        geo_features = X_encoded[:,1:] # (I*R*M, 15)

        return density, geo_features
    
    def forwardColour(self, geo_features, D):
        """
        Forward pass of the map
        Args:
            geo_features: batch of geometric features from point X; torch.tensor (I*R*M, 15)
            D: batch of directions, normalized vectors indicating direction of view; torch.tensor (I*R, D)
        Returns:
            colour: batch of colours from point X and with viewing direction D; torch.tensor (I*R*M, 3)
        """
        # encode direction
        D_encoded = self._encodeDirection(D) # (I*R*M, 2*f*D)
        D_encoded = torch.cat((geo_features, D_encoded), dim=1)

        # colour prediction
        colour = self.relu(self.colour_lin1(D_encoded))
        colour = self.relu(self.colour_lin2(colour))
        colour = self.sigmoid(self.colour_lin3(colour))

        return colour
    
    def _encodePosition(self, X):
        """
        Encode position vector X
        Args:
            X: batch of points; torch.tensor (I*R*M, D)
        Returns:
            X_encoded: torch.tensor (I*R*M, L*F)
        """
        # concatenate encoding from every layer
        X_encoded = torch.empty((X.shape[0], 0), dtype=torch.float32).to(self.args.device)
        for grid in self.grids:
            X_encoded = torch.cat((X_encoded, grid.forward(X).to(dtype=torch.float32)), dim=1)

        return X_encoded

    def _encodeDirection(self, D):
        """
        Encode direction vector D
        Args:
            D: batch of directions, normalized vectors indicating direction of view; torch.tensor (I*R, D)
        Returns:
            D_encoded: torch.tensor (I*R*M, 2*f*D)
        """
        # encode direction
        D_encoded = torch.empty((D.shape[0], 0), dtype=torch.float32).to(self.args.device)
        for i in range(self.args.f):
            D_encoded = torch.cat((D_encoded, torch.sin(2**i * torch.pi * D).to(dtype=torch.float32)), dim=1)
            D_encoded = torch.cat((D_encoded, torch.cos(2**i * torch.pi * D).to(dtype=torch.float32)), dim=1)

        # extand direction to match positions dimension
        D_encoded = D_encoded.repeat(self.args.M, 1) # (I*R*M, 2*f*D)

        return D_encoded

    






        


        