import numpy as np
import torch
import torch.nn as nn

from args import Args
from map import Map

class Trainer():
    def __init__(self, args:Args) -> None:
        self.args = args

        self.map = Map()

        self.optimizer = torch.optim.Adam(self.map.parameters(), lr=self.args.lr)

    def train(self, data_loader):
        
        for data in data_loader:
            X, D, ray_col_gt = data

            # forward pass through map
            sample_dens, sample_col = self.map(X, D)

            # estimate colour from sample densities and sample colours
            ray_col = self._colourEstimation(sample_dens, sample_col, sample_dist)

            # compute colour loss
            loss = self._colourLoss(ray_col, ray_col_gt)

            # backpropagate
            loss.backward()
            self.optimizer.step()




    def _colourEstimation(self, sample_dens, sample_col, sample_dist):
        """
        Estimate colour from density and colour
        Args:
            density: predicted density, torch.tensor (N, 1)
            colour: predicted colour, torch.tensor (N, 3)
            sample_dist: distance to sample point, torch.tensor (N, 1)
        Returns:
            ray_col: estimated colour of each ray, torch.tensor (R, 3)
        """
        # reshape from batch to nb. of rays times nb. of samples
        density = sample_dens.reshape(self.args.R, self.args.M) # (R, M)
        colour = sample_col.reshape(self.args.R, self.args.M, 3) # (R, M, 3)
        dist = sample_dist.reshape(self.args.R, self.args.M) # (R, M)

        # compute transmittance
        dens_dist = density * dist # (R, M)
        transmittance = torch.exp( -torch.cumsum(dens_dist, dim=1) ) # (R, M)

        # compute colour estimation
        ray_col = torch.einsum('rm,rm,rmc->rc', transmittance, (1-torch.exp(-dens_dist)), colour) # (R, 3)

        return ray_col


    def _colourLoss(self, ray_col, ray_col_gt):
        """
        Compute colour loss
        Args:
            ray_col: predicted colour, torch.tensor (R, 3)
            ray_col_gt: ground truth colour, torch.tensor (R, 3)
        Returns:
            loss: colour loss, torch.tensor (1)
        """
        loss = torch.mean( (ray_col - ray_col_gt)**2 )
        return loss

