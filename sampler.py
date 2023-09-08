import numpy as np
import torch

from args import Args
from plotter import Plotter
from dataloader import DataLoader


class Sampler():
    def __init__(self, args:Args) -> None:
        self.args = args

        # random number generator
        self.rng = np.random.default_rng(self.args.rand_seed) 
        self.nb_batches = self.args.load_nb_imgs["test"] // self.args.I # nb of batches per epoch     

    def iterData(self, imgs, rays):
        """
        Iterate once through entire dataset (1 epoch). Yield for every batch I images,
        R rays per image and M points per ray: samples per batch = I*R*M.
        Args:
            imgs: images; torch.tensor (N, H, W, 4)
            rays: rays; torch.tensor (N, ro+rd, H, W, 3)
        Yields:
            batch: batch of data (points, directions, colours_gt); tuple of torch.tensor (I*R*M, 3), (I*R, 3), (I*R, 4)
        """
        # calculate nb. batches
        self.nb_batches = imgs.shape[0] // self.args.I

        if self.args.verb_sampling:
            print(f"Start iterating data: nb. batches: {self.nb_batches}")

        # # shuffle data
        # permutation = self.rng.permutation(imgs.shape[0])
        # imgs = imgs[permutation]
        # rays = rays[permutation]

        for step in range(self.nb_batches):
            # extract images and rays of batch
            idx_range = range(step*self.args.I, (step+1)*self.args.I)
            img_batch = imgs[idx_range]
            ray_batch = rays[idx_range]

            # sample rays, points and colours
            ray_coord = self._sampleRays(img_batch) # (I*R, 3)
            points, directions = self.samplePoints(ray_batch, ray_coord) # (I*R*M, 3), (I*R, 3)
            colours_gt = self._sampleColours(img_batch, ray_coord) # (I*R, 4)

            # create data batch
            batch = (points, directions, colours_gt) # (I*R*M, 3), (I*R, 3), (I*R, 4)

            if self.args.verb_sampling:
                print(f"Sampled batch {step+1}/{self.nb_batches}, \
                      points shape: {points.shape}, directions shape: {directions.shape}, colours_gt shape: {colours_gt.shape}")

            yield batch

        # TODO: process left over images too

    def samplePoints(self, ray_batch, ray_coord):
        """
        Sample points from ray batch.
        Args:
            ray_batch: batch of rays; torch.tensor (I, ro+rd, H, W, 3)
            ray_coord: sampled ray coordinates [img. index, height coord., width coord.]; np.array (I*R, 3)
        Returns:
            points: sampled position on ray; torch.tensor (I*R*M, 3)
            directions: sample dirction of rays; torch.tensor (I*R, 3)
        """
        # extract ray origins and directions
        ray_origins = ray_batch[ray_coord[:,0], 0, ray_coord[:,1], ray_coord[:,2], :] # (I*R, 3)
        ray_directions = ray_batch[ray_coord[:,0], 1, ray_coord[:,1], ray_coord[:,2], :] # (I*R, 3)

        # calculate max. step that one can take on ray while keeping inside of [-1,1]**3
        ray_sign = torch.sign(ray_directions) # (I*R, 3), mirror problem if direction is negative
        t_max = torch.min( (1 - ray_sign*ray_origins) / (ray_sign*ray_directions), dim=1)[0] # (I*R,)

        # calc. intersection of ray with cube [-1,1]**3
        ray_max = ray_origins + t_max[:, None] * ray_directions

        # sample points on rays     
        points = np.linspace(ray_origins.detach().numpy(), ray_max.detach().numpy(), self.args.M) # (M, I*R, 3)
        points = np.transpose(points, (1, 0, 2)).reshape(self.args.I*self.args.R*self.args.M, 3) # (I*R*M, 3)
        points = torch.tensor(points).to(self.args.device)

        return points, ray_directions

    def _calcNbBatches(self, N):
        """
        Calculate nb. of batches per epoch.
        Args:
            N: nb. of images in dataset; int
        """
        if self.nb_batches is not None:
            return

        self.nb_batches = N // self.args.I
        if self.nb_batches == 0:
            raise ValueError(f"Batch size {self.args.I} is larger than dataset size {N}")

    def _sampleRays(self, img_batch):
        """
        Sample rays from image batch.
        Args:
            img_batch: batch of images; torch.tensor (I, H, W, 4)
        Returns:
            ray_coord: sampled ray coordinates [img. index, height coord., width coord.]; np.array (I*R, 3)
        """
        # sample ray coordinates
        height_coord = self.rng.integers(low=0, high=img_batch.shape[1], size=self.args.I*self.args.R)
        width_coord = self.rng.integers(low=0, high=img_batch.shape[2], size=self.args.I*self.args.R)
        img_idxs = np.arange(self.args.I).repeat(self.args.R)

        # stack coordinates
        ray_coord = np.stack([img_idxs, height_coord, width_coord], axis=-1)

        return ray_coord
    
    def _sampleColours(self, imgs, ray_coord):
        """
        Sample colours from image batch corresponding to rays.
        Args:
            imgs: batch of images; torch.tensor (I, H, W, 4)
            ray_coord: sampled ray coordinates [img. index, height coord., width coord.]; np.array (I*R, 3)
        Returns:
            colours_gt: sampled colours; torch.tensor (I*R, 4)
        """
        return imgs[ray_coord[:,0], ray_coord[:,1], ray_coord[:,2], :]


def test_iterData():
    args = Args()
    dl = DataLoader(args=args)
    plot = Plotter(args=args)
    sampler = Sampler(args=args)

    # load data and move it to current device
    imgs, rays = dl.loadData()
    imgs = {split: torch.tensor(data).to(args.device) for split, data in imgs.items()}
    rays = {split: torch.tensor(data).to(args.device) for split, data in rays.items()}
    
    for i, batch in enumerate(sampler.iterData(imgs, rays)):
        # unpack batch
        points, directions, colours_gt = batch # (I*R*M, 3), (I*R, 3), (I*R, 4)

        # plot camera poses of batch
        ray_batch = rays["test"][i*args.I:(i+1)*args.I]
        ray_batch = {"test": ray_batch.to("cpu").detach().numpy()}
        plot.cameraPositions(ray_batch)
        plot.cameraFoV(ray_batch)
        plot.cameraRays(points.to("cpu").detach().numpy(), directions.to("cpu").detach().numpy(), ray_batch)
        plot.showPlot()

    

if __name__ == "__main__":
    test_iterData()