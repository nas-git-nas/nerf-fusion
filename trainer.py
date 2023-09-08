import numpy as np
import torch
import torch.nn as nn
import time
from alive_progress import alive_bar

from args import Args
from map import Map
from dataloader import DataLoader
from sampler import Sampler
from plotter import Plotter





class Trainer():
    def __init__(self, args:Args) -> None:
        self.args = args

        self.map = Map(args=args)
        self.data_loader = DataLoader(self.args)
        self.sampler = Sampler(self.args)

        # optimizer
        self.optimizer = torch.optim.Adam(self.map.parameters(), lr=self.args.lr)

        # logging
        self.losses_batch = [] # loss per batch
        self.losses_test = [] # loss per test image

        # plotter
        self.plot = Plotter(self.args)

    def train(self):

        # load data and move it to current device
        imgs, rays = self.data_loader.loadData(splits="train")
        imgs = torch.tensor(imgs["train"]).to(self.args.device)
        rays = torch.tensor(rays["train"]).to(self.args.device)

        if self.args.verb_training:
            print(f'Start training \t--- \tnb. epochs: {self.args.nb_epochs}, \
                  nb. imgs: {self.args.I}, nb. rays per img: {self.args.R}, nb. samples per ray: {self.args.M}')
            time_total = time.time()
            time_epoch = time.time()

        for epoch in range(self.args.nb_epochs):
            
            for j, batch in enumerate(self.sampler.iterData(imgs, rays)):
                # unpack batch
                points, directions, colours_gt = batch # (I*R*M, 3), (I*R, 3), (I*R, 4)

                # forward pass through map
                sample_dens, sample_col = self.map(X=points, D=directions) # (I*R*M,), (I*R*M, 3)

                # estimate colour from sample densities and sample colours
                colours = self._colourEstimation(sample_dens, sample_col, points) # (I*R, 3)

                # compute colour loss
                loss = self._colourLoss(colours, colours_gt)
                self.losses_batch.append(loss.item())

                # backpropagate
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # # progress bar for one epoch
                # with alive_bar(self.sampler.nb_batches, bar = 'bubbles') as bar:
                    # if self.args.verb_training:
                    #     bar()

            if self.args.verb_training:
                print(f"Epoch {epoch+1}/{self.args.nb_epochs}, \tloss: {np.mean(self.losses_batch[-self.sampler.nb_batches:]):.4f}, \
                      \ttotal time; {time.time()-time_total:.2f}s, \tavg. batch time: {(time.time()-time_epoch)/self.sampler.nb_batches:.2f}s")
                time_epoch = time.time()

        if self.args.verb_training:
            print(f'End training \t--- \ttotal time: {time.time()-time_total:.2f}s')

    def test(self):
        # load data and move it to current device
        imgs, rays = self.data_loader.loadData(splits="test")
        imgs = torch.tensor(imgs["test"]).to(self.args.device) # (N, H, W, 4)
        rays = torch.tensor(rays["test"]).to(self.args.device) # (N, ro+rd, H, W, 3)

        if self.args.verb_training:
            print(f'Start testing \t--- \tnb. epochs: {self.args.nb_epochs}, \
                  nb. imgs: {self.args.I}, nb. rays per img: {self.args.R}, nb. samples per ray: {self.args.M}')
            time_total = time.time()
            time_epoch = time.time()
            time_batch = time.time()

        imgs_est = []
        for i in range(imgs.shape[0]):
            # height and width of image
            H = imgs.shape[1]
            W = imgs.shape[2]

            coord = torch.meshgrid((torch.arange(H), torch.arange(W))) # [(H,W), (H,W))]
            ray_coord = torch.stack((torch.zeros(H*W), coord[0].flatten(), coord[1].flatten()), dim=1) # (3, H*W)
            ray_batch = rays[i].reshape(1, 2, H, W, 3) # (1, ro+rd, H, W, 3)
            points, directions = self.sampler.samplePoints(ray_batch=ray_batch, ray_coord=ray_coord) # (H*W*M, 3), (H*W, 3)

            colours_gt = imgs[i].reshape(1,H*W, 4) # (H*W, 4)

            # forward pass through map
            sample_dens, sample_col = self.map(X=points, D=directions) # (H*W*M,), (H*W*M, 3)

            # estimate colour from sample densities and sample colours
            colours = self._colourEstimation(sample_dens, sample_col, points) # (H*W, 3)
            imgs_est.append(colours.reshape(H,W,3).detach().cpu().numpy())

            # compute colour loss
            loss = self._colourLoss(colours, colours_gt)
            self.losses_test.append(loss.item())

        imgs_gt = [imgs[i,:,:,:3].detach().cpu().numpy() for i in imgs.shape[0]]
        self.plot.showTestImgs(imgs_gt, imgs_est, self.losses_test)

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
        points = points.reshape(self.args.I*self.args.R, self.args.M, 3) # (I*R, M, 3)

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
            colours_gt: ground truth colour, torch.tensor (I*R, 4)
        Returns:
            loss: colour loss, torch.tensor (1)
        """
        loss = torch.mean( (colours - colours_gt[:,:3])**2 )
        return loss





def test_trainer():
    args = Args()
    trainer = Trainer(args)
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    test_trainer()