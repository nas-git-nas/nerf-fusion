import torch

class Args():
    def __init__(self) -> None:
        # hardware
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"
        self.device = torch.device(dev)


        # general
        self.rand_seed = 42 # random seed


        self.D = 3 # nb of dimensions
        self.L = 3 # nb of grid layers
        self.T = 2**14 # hash table length
        self.F = 2 # nb of features, hash table depth
        self.f = 4 # order of frequency encoding 
        
        self.res_min = 16
        self.res_max = 64

        self.M = 128 # nb of samples per ray
        self.R = 32 # nb of rays per image
        self.I = 1 # nb of images per batch
        self.N = None # dataset size (total number of images)

        self.data_dir = '../data/nerf_synthetic/lego'
        self.img_half_res = True

        # training
        self.nb_epochs = 1


        # plotting
        self.nb_camera_poses = 30