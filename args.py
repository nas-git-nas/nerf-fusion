import torch
import os
import shutil
import pandas as pd

from datetime import datetime

class Args():
    def __init__(self) -> None:
        # general
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"
        self.device = torch.device(dev)
        self.rand_seed = 42 # random seed
        self.debug = True

        # verbose
        self.verb_training = True
        self.verb_sampling = False

        # data
        if self.device == torch.device("cuda:0"):
            self.data_dir = '/media/scratch1/schmin/data/nerf_synthetic/lego'
        else:
            self.data_dir = '../data/nerf_synthetic/lego'
        self.img_half_res = True
        self.load_nb_imgs = {"train": 100, "test": 2, "val": 0}

        # model
        self.nb_epochs_per_checkpoint = 2
        self.checkpoint_path = None #'./models/20230919_0549/model.pt'
        t = datetime.now()
        dir_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M")
        self.model_path = os.path.join("models", dir_name)
        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)
        os.mkdir(self.model_path)

        # logging
        self.log_df = pd.DataFrame(columns=['batch', 'loss_batch', 'lr'])
        self.log_df.to_csv(os.path.join(self.model_path, "log.csv"), index=False)

        # map (multi-layer hash encoding)
        self.D = 3 # nb of dimensions
        self.L = 1 #6 # nb of grid layers
        self.T = 512**3 #2**14 # hash table length
        self.F = 2 # nb of features, hash table depth
        self.f = 4 # order of frequency encoding       
        self.res_min = 16
        self.res_max = 512     

        # training
        self.nb_epochs = 60
        self.lr = 5e-5
        self.M = 256 # nb of samples per ray
        self.R = 4096 # nb of rays per image
        if self.device == torch.device("cuda:0"):
            self.I = 2 # nb of images per batch
        else:
            self.I = 2
        self.N = None # dataset size (total number of images)

        # plotting
        self.plot_nb_I = 30
        self.plot_nb_R = 10
        self.plot_nb_M = 30