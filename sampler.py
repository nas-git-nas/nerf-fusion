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
        self.nb_batches = None 

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

        # clip positions to account for numerical errors
        if self.args.debug:
            if not ((torch.max(rays[:,0,:,:,:])-1<0.000001) and (torch.min(rays[:,0,:,:,:])+1>-0.000001)):
                print(f"ERROR: dataloader._scaleCoords: positions are out of bounds [-1,1], max: {torch.max(rays[:,0,:,:,:])}, min: {torch.min(rays[:,0,:,:,:])}")
        rays[:,0,:,:,:] = torch.clamp(rays[:,0,:,:,:], min=-1, max=1)

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

            # points, directions = self._samplePoints(ray_batch, ray_coord) # (I*R*M, 3), (I*R, 3)
            points, directions = self._samplePointsExp(ray_batch, ray_coord) # (I*R*M, 3), (I*R, 3)


            colours_gt = self._sampleColours(img_batch, ray_coord) # (I*R, 4)

            # create data batch
            batch = (points, directions, colours_gt) # (I*R*M, 3), (I*R, 3), (I*R, 4)

            if self.args.verb_sampling:
                print(f"Sampled batch {step+1}/{self.nb_batches}, \
                      points shape: {points.shape}, directions shape: {directions.shape}, colours_gt shape: {colours_gt.shape}")

            yield batch

        # TODO: process left over images too

    def iterTestData(self, imgs, rays, downsample_factor=1):
        """
        Iterate once through all test imgs and return a batch for every image.
        Args:
            imgs: images; torch.tensor (N, H, W, 4)
            rays: rays; torch.tensor (N, ro+rd, H, W, 3)
        Yields:
            batch: batch of data (points, directions, colours_gt); tuple of torch.tensor (I*R*M, 3), (I*R, 3), (I*R, 4)
        """
        # downsample data
        coord_H = np.arange(0, imgs.shape[1], step=downsample_factor, dtype=np.int32) # (h)
        coord_W = np.arange(0, imgs.shape[2], step=downsample_factor, dtype=np.int32) # (w)
        h = len(coord_H) # height of downsampled img
        w = len(coord_W) # width of downsampled img
        imgs = imgs[:,coord_H,:,:] # (N, h, W, 4)
        imgs = imgs[:,:,coord_W,:] # (N, h, w, 4)
        rays = rays[:,:,coord_H,:,:] # (N, ro+rd, h, W, 3)
        rays = rays[:,:,:,coord_W,:] # (N, ro+rd, h, w, 3)

        # get ray coordinates
        coord = np.meshgrid(np.arange(h), np.arange(w)) # [(h, w), (h, w)]
        ray_coord = np.stack((np.zeros(h*w), coord[0].flatten(), coord[1].flatten()), axis=1).astype(np.int32) # (h*w, 3)

        # calculate points and direction for each ray
        for i in range(imgs.shape[0]):
            # sample points along ray
            ray_batch = rays[i].reshape(1, 2, h, w, 3) # (1, ro+rd, h, w, 3)
            points, directions = self._samplePoints(ray_batch=ray_batch, ray_coord=ray_coord) # (h*w*M, 3), (h*w, 3)

            colours_gt = imgs[i].reshape(h*w, 4) # (h*w, 4)

            # create data batch
            batch = (points, directions, colours_gt, h, w) # (h*w*M, 3), (h*w, 3), (h*w, 4), int, int

            yield batch


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
        ray_origins = ray_batch[ray_coord[:,0], 0, ray_coord[:,1], ray_coord[:,2], :] # (I*R, 3)
        ray_directions = ray_batch[ray_coord[:,0], 1, ray_coord[:,1], ray_coord[:,2], :] # (I*R, 3)

        # calculate max. step that one can take on ray while keeping inside of [-1,1]**3
        ray_sign = torch.sign(ray_directions) # (I*R, 3), mirror problem if direction is negative
        t_max = torch.min( (1 - ray_sign*ray_origins) / (ray_sign*ray_directions), dim=1)[0] # (I*R,)

        # calc. intersection of ray with cube [-1,1]**3
        ray_max = ray_origins + t_max[:, None] * ray_directions

        # sample points on rays     
        points = np.linspace(ray_origins.detach().cpu().numpy(), ray_max.detach().cpu().numpy(), self.args.M) # (M, I*R, 3)
        points = np.transpose(points, (1, 0, 2)).reshape(ray_coord.shape[0]*self.args.M, 3) # (I*R*M, 3)
        points = torch.tensor(points).to(self.args.device)

        return points, ray_directions
    
    def _samplePointsExp(self, ray_batch, ray_coord):
        """
        Sample points from ray batch exponentially around image depth.
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

        # calc. step that leads to closest point to origin
        t_closest = -torch.sum(ray_origins * ray_directions, axis=1) / torch.sum(ray_directions**2, axis=1) # (I*R,)
        t_closest = t_closest.to("cpu").detach().numpy() # (I*R,)
        t_closest = np.clip(t_closest, a_min=0, a_max=t_max) # (I*R,)

        # sample depths of ray
        depths = np.repeat(t_closest.reshape(-1,1), self.args.M, axis=1) # (I*R, M)
        depths = self.rng.normal(loc=depths, scale=0.4, size=(ray_coord.shape[0], self.args.M)) # (I*R, M)
        depths = np.clip(depths, a_min=0, a_max=np.repeat(t_max.reshape(-1,1), self.args.M, axis=1)) # (I*R, M)
        depths = np.sort(depths, axis=1) # (I*R, M)
        depths = torch.tensor(depths.flatten()).to(self.args.device) # (I*R*M,)

        # calc. position of sampled points
        points = torch.repeat_interleave(ray_origins, self.args.M, dim=0) \
                 + depths[:,None] * torch.repeat_interleave(ray_directions, self.args.M, dim=0) # (I*R*M, 3)

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
    imgs, rays = dl.loadData(splits=["train"])
    imgs = torch.tensor(imgs["train"], dtype=torch.float32).to(args.device)
    rays = torch.tensor(rays["train"], dtype=torch.float32).to(args.device)
    
    for i, batch in enumerate(sampler.iterData(imgs, rays)):
        # unpack batch
        points, directions, colours_gt = batch # (I*R*M, 3), (I*R, 3), (I*R, 4)

        # plot camera poses of batch
        ray_batch = rays[i*args.I:(i+1)*args.I].detach().cpu().numpy() # (I, ro+rd, H, W, 3)
        # plot.cameraPositions(ray_batch)
        plot.cameraFoV(ray_batch)
        plot.cameraRays(points.to("cpu").detach().numpy(), directions.to("cpu").detach().numpy(), ray_batch, colours_gt.to("cpu").detach().numpy())
        plot.showPlot()

if __name__ == "__main__":
    test_iterData()