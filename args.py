

class Args():
    def __init__(self) -> None:
        self.D = 3 # nb of dimensions
        self.L = 3 # nb of grid layers
        self.T = 2**14 # hash table length
        self.F = 2 # nb of features, hash table depth
        self.f = 4 # order of frequency encoding 
        
        self.res_min = 16
        self.res_max = 64

        self.M = 128 # nb of samples per ray
        self.R = 1 # nb of rays per batch
        self.N = self.M * self.R # nb of samples per batch