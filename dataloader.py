
# https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec#:~:text=The%204x4%20transformation%20matrix%20that,as%20the%20camera%20extrinsic%20matrix.


# https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py



import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

from args import Args
from plotter import Plotter

class DataLoader():
    def __init__(self, args:Args) -> None:
        self.args = args

    def loadData(self, splits=['train', 'val', 'test']):
        """
        Load images and poses from the data directory and calculate rays for every image and pixel.
        Args:
            splits: list of splits to load; list of strings
        Returns:
            imgs: images; dict of numpy array of shape (N, H, W, 4)
            rays: rays; dict of numpy array of shape (N, ro+rd, H, W, 3)
        """
        # load images and poses
        splits, imgs, poses, render_poses, [H, W, focal], i_split = self._loadBlenderData(splits=splits)

        # calculate rays
        K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])
        rays = np.stack([self._getRays(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]

        # split data into train, val, test
        imgs_dict = { s:imgs[i_split[s]] for s in splits}
        rays_dict = { s:rays[i_split[s]] for s in splits}

        return imgs_dict, rays_dict

    def _loadBlenderData(self, splits):
        """
        Args:
            splits: list of splits to load; list of strings
        """

        # remove split if now imgs are loaded
        # remove_splits = []
        # for s in splits:
        #     if self.args.load_nb_imgs[s] == 0:
        #         remove_splits.append(s)
        # for s in remove_splits:
        #     splits.remove(s)

        metas = {}
        for s in splits:
            with open(os.path.join(self.args.data_dir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []

            # if s=='train' or testskip==0:
            #     skip = 1
            # else:
            #     skip = testskip

            # calculate skip step size
            step = len(meta['frames']) // self.args.load_nb_imgs[s]
                
            for frame in meta['frames'][::step]:
                fname = os.path.join(self.args.data_dir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            
            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
        
        i_split = { s:np.arange(counts[i], counts[i+1]) for i, s in enumerate(splits) }
        
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)

        # # TODO: implement this
        # # keep only 3 pictures
        # imgs = imgs[:3]
        # poses = poses[:3]
        # i_split = [np.arange(3), np.arange(3), np.arange(3)]
        
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        # render_poses = torch.stack([self._poseSpherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        render_poses = None
        
        if self.args.img_half_res:
            H, W, focal, imgs = self._halfResolution(H, W, focal, imgs)
           
        return splits, imgs, poses, render_poses, [H, W, focal], i_split
    
    def _halfResolution(self, H, W, focal, imgs):
            H = H//2
            W = W//2
            focal = focal/2.

            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
            
            return H, W, focal, imgs
       
    def _poseSpherical(self, theta, phi, radius):
        c2w = self._transitionT(radius)
        c2w = self._rotatePhi(phi/180.*np.pi) @ c2w
        c2w = self._rotateTheta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    
    def _transitionT(self, t):
        return  torch.Tensor([
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,t],
                    [0,0,0,1]]
                ).float()

    def _rotatePhi(self, phi):
        return  torch.Tensor([
                    [1,0,0,0],
                    [0,np.cos(phi),-np.sin(phi),0],
                    [0,np.sin(phi), np.cos(phi),0],
                    [0,0,0,1]]
                ).float()

    def _rotateTheta(self, theta):
        return  torch.Tensor([
                    [np.cos(theta),0,-np.sin(theta),0],
                    [0,1,0,0],
                    [np.sin(theta),0, np.cos(theta),0],
                    [0,0,0,1]]
                ).float()

    def _getRays(self, H, W, K, c2w):
        """
        Get ray origins and directions for all pixels in an image.
        Args:
            H, W: Image height and width; int
            K: Camera intrinsic matrix; np.array of shape (3, 3)
            c2w: Camera-to-world matrix; np.array of shape (4, 4)
        Returns:
            rays_o: the ray origins; Tensor of shape (H, W, 3)
            rays_d: the ray directions; Tensor of shape (H, W, 3)
        """
        if torch.is_tensor(c2w):
            return self._getRaysTorch(H, W, K, c2w)
        else:
            return self._getRaysNp(H, W, K, c2w)

    def _getRaysTorch(self, H, W, K, c2w):
        """
        Get ray origins and directions for all pixels in an image.
        Args:
            H, W: Image height and width; int
            K: Camera intrinsic matrix; np.array of shape (3, 3)
            c2w: Camera-to-world matrix; np.array of shape (4, 4)
        Returns:
            rays_o: the ray origins; Tensor of shape (H, W, 3)
            rays_d: the ray directions; Tensor of shape (H, W, 3)
        """
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        return rays_o, rays_d

    def _getRaysNp(self, H, W, K, c2w):
        """
        Get ray origins and directions for all pixels in an image.
        Args:
            H, W: Image height and width; int
            K: Camera intrinsic matrix; np.array of shape (3, 3)
            c2w: Camera-to-world matrix; np.array of shape (4, 4)
        Returns:
            rays_o: the ray origins; Tensor of shape (H, W, 3)
            rays_d: the ray directions; Tensor of shape (H, W, 3)
        """
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
        return rays_o, rays_d



def test_dataloader():
    args = Args()
    dl = DataLoader(args=args)
    plot = Plotter(args=args)

    imgs, rays = dl.loadData()
    plot.cameraPoses(rays)

    

if __name__ == "__main__":
    test_dataloader()

