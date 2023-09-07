import numpy as np
import torch
import torch.nn as nn

from args import Args
from map import Map
from dataloader import DataLoader
from sampler import Sampler





class Trainer():
    def __init__(self, args:Args) -> None:
        self.args = args

        self.map = Map()

        self.optimizer = torch.optim.Adam(self.map.parameters(), lr=self.args.lr)

        self.data_loader = DataLoader(self.args)
        self.sampler = Sampler(self.args)

    def train(self):

        # load data and move it to current device
        imgs, rays = self.data_loader.loadData()
        imgs = {split: torch.tensor(data).to(self.args.device) for split, data in imgs.items()}
        rays = {split: torch.tensor(data).to(self.args.device) for split, data in rays.items()}

        for epoch in range(self.args.nb_epochs):
        
            for batch in self.sampler.iterData(imgs, rays):
                # unpack batch
                points, directions, colours_gt = batch # (I*R*M, 3), (I*R, 3), (I*R, 4)

                # forward pass through map
                sample_dens, sample_col = self.map(X=points, D=directions) # (I*R*M,), (I*R*M, 3)

                # estimate colour from sample densities and sample colours
                colours = self._colourEstimation(sample_dens, sample_col, points) # (I*R, 3)

                # compute colour loss
                loss = self._colourLoss(colours, colours_gt)

                # backpropagate
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def _colourEstimation(self, sample_dens, sample_col, points):
        """
        Estimate colour from density and colour
        Args:
            density: predicted density, torch.tensor (I*R*M, 1)
            colour: predicted colour, torch.tensor (I*R*M, 3)
            points: sampled points, torch.tensor (I*R*M, 3)
        Returns:
            colours: estimated colour of each ray, torch.tensor (I*R, 3)
        """
        # reshape from batch to nb. of rays times nb. of samples
        density = sample_dens.reshape(self.args.I*self.args.R, self.args.M) # (I*R, M)
        colour = sample_col.reshape(self.args.I*self.args.R, self.args.M, 3) # (I*R, M, 3)
        points = points.reshape(self.args.I*self.args.R, self.args.M) # (I*R, M, 3)

        # compute distance between points
        next_points = torch.roll(points, shifts=-1, dims=1) # (I*R, M, 3)
        dist = torch.norm(next_points - points, dim=2) # (I*R, M)
        dist[:,-1] = 0 # last point has no successor

        # compute transmittance
        dens_dist = density * dist # (I*R, M)
        transmittance = torch.exp( -torch.cumsum(dens_dist, dim=1) ) # (I*R, M)

        # compute colour estimation
        colours = torch.einsum('rm,rm,rmc->rc', transmittance, (1-torch.exp(-dens_dist)), colour) # (I*R, 3)

        return colours


    def _colourLoss(self, colours, colours_gt):
        """
        Compute colour loss
        Args:
            colours: predicted colour, torch.tensor (I*R, 3)
            colours_gt: ground truth colour, torch.tensor (I*R, 3)
        Returns:
            loss: colour loss, torch.tensor (1)
        """
        loss = torch.mean( (colours - colours_gt)**2 )
        return loss

