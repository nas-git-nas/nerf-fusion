import numpy as np
import torch

from args import Args


class Sampler():
    def __init__(self, args:Args) -> None:
        self.args = args

        # random number generator
        self.rng = np.random.default_rng(self.args.rand_seed)      

    def iterData(self, imgs, rays):
        """
        Iterate once through entire dataset (1 epoch). Yield for every batch I images,
        R rays per image and M points per ray: samples per batch = I*R*M.
        Args:
            imgs: images; dict of torch.tensor (N, H, W, 4)
            rays: rays; dict of torch.tensor (N, ro+rd, H, W, 3)
        Yields:
            batch: batch of data (points, directions, colours_gt); tuple of torch.tensor (I*R*M, 3), (I*R, 3), (I*R, 4)
        """

        # # shuffle data
        # permutation = self.rng.permutation(imgs.shape[0])
        # imgs = imgs[permutation]
        # rays = rays[permutation]

        nb_steps = imgs.shape[0] // self.args.I
        for step in range(nb_steps):
            # extract images and rays of batch
            idx_range = range(step*self.args.I, (step+1)*self.args.I)
            img_batch = imgs["test"][idx_range]
            ray_batch = rays["test"][idx_range]

            # sample rays, points and colours
            ray_coord = self._sampleRays(img_batch) # (I*R, 3)
            points, directions = self._samplePoints(ray_batch, ray_coord) # (I*R*M, 3), (I*R, 3)
            colours_gt = self._sampleColours(img_batch, ray_coord) # (I*R, 4)

            # create data batch
            batch = (points, directions, colours_gt) # (I*R*M, 3), (I*R, 3), (I*R, 4)

            yield batch

        # TODO: process left over images too

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
        ray_coord = torch.stack([img_idxs, height_coord, width_coord], dim=-1)

        return ray_coord
    
    def _samplePoints(self, ray_batch, ray_coord):
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
        ray_origins = ray_batch[ray_coord[0], 0, ray_coord[1], ray_coord[2], :] # (I*R, 3)
        ray_directions = ray_batch[ray_coord[0], 1, ray_coord[1], ray_coord[2], :] # (I*R, 3)

        # calculate max. step that one can take on ray while keeping inside of [-1,1]**3
        ray_sign = torch.sign(ray_directions) # (I*R, 3), mirror problem if direction is negative
        t_max = torch.min( (1 - ray_sign*ray_origins) / (ray_sign*ray_directions), dim=1) # (I*R,)

        # calc. intersection of ray with cube [-1,1]**3
        ray_max = ray_origins + t_max[:, None] * ray_directions

        # sample points on rays     
        points = np.linspace(ray_origins.detach().to_numpy(), ray_max.detach().to_numpy(), self.args.M) # (M, I*R, 3)
        points = np.transpose(points, (1, 0, 2)).reshape(self.args.I*self.args.R*self.args.M, 3) # (I*R*M, 3)
        points = torch.tensor(points).to(self.args.device)

        return points, ray_directions
    
    def _sampleColours(self, imgs, ray_coord):
        """
        Sample colours from image batch corresponding to rays.
        Args:
            imgs: batch of images; torch.tensor (I, H, W, 4)
            ray_coord: sampled ray coordinates [img. index, height coord., width coord.]; np.array (I*R, 3)
        Returns:
            colours_gt: sampled colours; torch.tensor (I*R, 4)
        """
        return imgs[ray_coord[0], ray_coord[1], ray_coord[2], :]

