import torch

class Args():
    def __init__(self) -> None:
        # general
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"
        self.device = torch.device(dev)
        self.rand_seed = 42 # random seed

        # verbose
        self.verb_training = True
        self.verb_sampling = False

        # data
        self.data_dir = '../data/nerf_synthetic/lego'
        self.img_half_res = True
        self.load_nb_imgs = {"train": 100, "test": 2, "val": 0}

        # map (multi-layer hash encoding)
        self.D = 3 # nb of dimensions
        self.L = 3 # nb of grid layers
        self.T = 2**14 # hash table length
        self.F = 2 # nb of features, hash table depth
        self.f = 4 # order of frequency encoding       
        self.res_min = 16
        self.res_max = 64     

        # training
        self.nb_epochs = 10
        self.lr = 1e-4
        self.M = 128 # nb of samples per ray
        self.R = 4096 # nb of rays per image
        self.I = 2 # nb of images per batch
        self.N = None # dataset size (total number of images)

        # plotting
        self.plot_nb_I = 30
        self.plot_nb_R = 10
        self.plot_nb_M = 30