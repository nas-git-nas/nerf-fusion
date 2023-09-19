import numpy as np
import torch

from torch import nn

from args import Args


class Grid(nn.Module):
    def __init__(self, args:Args, layer:int) -> None:
        # init. nn.Module
        super(Grid, self).__init__()

        self.args = args
        self.rand_generator = torch.manual_seed(self.args.rand_seed)
        
        # calc. growth factor
        growth_factor = np.exp( (np.log(args.res_max) - np.log(args.res_min)) / (args.L - 1))
        self.res = int( (growth_factor**layer) * args.res_min )

        # create learnable gird
        self.hash_table = nn.Parameter(torch.rand((self.args.T, self.args.F), generator=self.rand_generator).to(self.args.device))

        # prime numbers for hash function
        self.primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.int64)
        
    def forward(self, X):
        """
        Forward pass of the grid
        Args:
            X: batch of points, torch.tensor (I*R*M, D)
        Returns:
            hash_vals: hashed and linear interpolated points, torch.tensor (I*R*M, F)
        """
        # get hash table indices
        with torch.no_grad():           
            hash_idxs = self._hashFunction(X) # (I*R*M, 2**D)

        # get hash table values 
        hash_vals = self.hash_table[hash_idxs] # (I*R*M, 2**D, F)

        # 3-linear interpolation
        hash_vals = self._linearInterpolation(X, hash_vals) # (I*R*M, F)

        return hash_vals # TODO: convert tensors to float32

    
    
    def _hashFunction(self, X):
        """
        Map hash cube indices to hash table indices
        Args:
            X: torch.tensor of shape (I*R*M, D)
        Returns:
            hash_idxs: hash table indices, tensor (I*R*M, 2**D)
        """
        # get cube indices
        cube_idxs = self._getCubeIdxs(X) # (I*R*M, 2**D, D)

        # do hash mapping
        hash_map = cube_idxs * self.primes # (I*R*M, 2**D, D)

        hash_idxs = hash_map[:,:,0]
        for i in range(1, self.args.D):
            hash_idxs = torch.bitwise_xor(hash_idxs, hash_map[:,:,i])
        hash_idxs = torch.remainder(hash_idxs, self.args.T)

        if self.args.debug:
            if not (hash_idxs>=0).all() and (hash_idxs<self.args.T).all():
                print(f"ERROR: grid._hashFunction: hash_idxs must be in [0,T-1]; hash_idxs: {hash_idxs}")
            # assert (hash_idxs>=0).all() and (hash_idxs<self.args.T).all(), "hash_idxs must be in [0,T-1]"

        return hash_idxs
    
    def _getCubeIdxs(self, X):
        """
        Get cube indices for each point in X
        Args:
            X: torch.tensor of shape (I*R*M, D)
        Returns:
            cube_indices: int torch.tensor of shape (I*R*M, 2**D, D)
        """
        if self.args.debug:
            if not X.shape[1]==self.args.D:
                print(f"ERROR: grid._getCubeIdxs: X must have D dimensions; X shape: {X.shape}, D: {self.args.D}")
            # assert X.shape[1]==self.args.D, "X must have D dimensions"

        # get grid indices
        X_scaled = self._convertPos2Index(X)
        X_floor = torch.floor(X_scaled).type(torch.int64)
        X_ceil = torch.ceil(X_scaled).type(torch.int64)
        X_ceil[X_ceil==X_floor] += 1

        # get cube indices
        cube_idxs = torch.empty((X.shape[0], np.power(2, self.args.D), self.args.D), dtype=torch.int64)
        cube_idxs[:, 0, :] = X_floor
        cube_idxs[:, 1, :] = torch.hstack((X_floor[:,0].reshape(-1,1), X_floor[:,1].reshape(-1,1), X_ceil[:,2].reshape(-1,1)))
        cube_idxs[:, 2, :] = torch.hstack((X_floor[:,0].reshape(-1,1), X_ceil[:,1].reshape(-1,1), X_floor[:,2].reshape(-1,1)))
        cube_idxs[:, 3, :] = torch.hstack((X_floor[:,0].reshape(-1,1), X_ceil[:,1].reshape(-1,1), X_ceil[:,2].reshape(-1,1)))
        cube_idxs[:, 4, :] = torch.hstack((X_ceil[:,0].reshape(-1,1), X_floor[:,1].reshape(-1,1), X_floor[:,2].reshape(-1,1)))
        cube_idxs[:, 5, :] = torch.hstack((X_ceil[:,0].reshape(-1,1), X_floor[:,1].reshape(-1,1), X_ceil[:,2].reshape(-1,1)))
        cube_idxs[:, 6, :] = torch.hstack((X_ceil[:,0].reshape(-1,1), X_ceil[:,1].reshape(-1,1), X_floor[:,2].reshape(-1,1)))
        cube_idxs[:, 7, :] = X_ceil

        return cube_idxs
    
    def _linearInterpolation(self, X, hash_vals):
        """
        D-linear interpolation of hash values
        Args:
            X: batch of points, tensor (I*R*M, D)
            hash_vals: hash values, tensor (I*R*M, 2**D, F)
        Returns:
            hash_vals: interpolated hash values, tensor (I*R*M, F)
        """
        if self.args.debug:
            if not hash_vals.shape[0]==X.shape[0]:
                print("ERROR: grid._linearInterpolation: hash_vals and X must have same number of points")
            if not hash_vals.shape[1]==np.power(2, self.args.D):
                print("ERROR: grid._linearInterpolation: hash_vals must have 2**D values")
            if not hash_vals.shape[2]==self.args.F:
                print("ERROR: grid._linearInterpolation: hash_vals must have F features")
            # assert hash_vals.shape[0]==X.shape[0], "hash_vals and X must have same number of points"
            # assert hash_vals.shape[1]==np.power(2, self.args.D), "hash_vals must have 2**D values"
            # assert hash_vals.shape[2]==self.args.F, "hash_vals must have F features"

        # get interpolation weights
        X_scaled = self._convertPos2Index(X)
        X_floor = torch.floor(X_scaled)
        weights = X_scaled - X_floor

        # interpolate along x
        w0 = weights[:,0].reshape(-1,1).repeat(1,self.args.F)
        p00 = hash_vals[:,0,:] * (1-w0) + hash_vals[:,4,:] * w0
        p01 = hash_vals[:,1,:] * (1-w0) + hash_vals[:,5,:] * w0
        p10 = hash_vals[:,2,:] * (1-w0) + hash_vals[:,6,:] * w0
        p11 = hash_vals[:,3,:] * (1-w0) + hash_vals[:,7,:] * w0

        # interpolate along y
        w1 = weights[:,1].reshape(-1,1).repeat(1,self.args.F)
        p0 = p00 * (1-w1) + p10 * w1
        p1 = p01 * (1-w1) + p11 * w1

        # interpolate along z
        w2 = weights[:,2].reshape(-1,1).repeat(1,self.args.F)
        hash_vals = p0 * (1-w2) + p1 * w2

        return hash_vals
    

    def _convertPos2Index(self, X):
        """
        Convert position from [-1,1] to [0,res-1]
        Args:
            X: torch.tensor of shape (I*R*M, D)
        Returns:
            X_scaled: torch.tensor of shape (I*R*M, D)
        """
        if self.args.debug:
            if not (X<=1.00001).all() and (X>=-1.00001).all():
                print(f"ERROR: grid._convertPos2Index: X must be in [-1,1]; X: {X}")
            # assert (X<=1.00001).all() and (X>=-1.00001).all(), f"X must be in [-1,1], min: {torch.min(X)}, max: {torch.max(X)}"

        # get grid indices
        X_scaled = (X + 1) / 2 # scale from [-1,1] to [0,1]
        X_scaled = X_scaled * (self.res-1) # scale from [0,1] to [0,res-1]

        return X_scaled



def test_getCubeIdxs():
    args = Args()
    grid = Grid(args, layer=0)

    X = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).to(args.device)
    cube_idxs = grid._getCubeIdxs(X)
    print(f"Resolution: {grid.res}")
    print(f"X: {X}")
    print(f"X scaled: {X*(grid.res-1)}")
    print(f"cube_idxs: {cube_idxs}")

def test_hashFunction():
    args = Args()
    grid = Grid(args, layer=0)

    X = torch.rand(1024,3).to(args.device)
    hash_idxs = grid._hashFunction(X)
    
    print(f"hash indices: {hash_idxs}")

def test_linearInterpolation():
    args = Args()
    grid = Grid(args, layer=0)

    X = torch.tensor([[0.1, 0.2, 0.3]]).to(args.device)
    cube_idxs = grid._getCubeIdxs(X)
    hash_vals_before = torch.concat((cube_idxs[:,:,1].reshape(X.shape[0], 2**args.D, 1), cube_idxs[:,:,2].reshape(X.shape[0], 2**args.D, 1)), dim=2)
    hash_vals = grid._linearInterpolation(X, hash_vals=hash_vals_before.to(args.device))
    
    print(f"cube indices: {cube_idxs}")
    print(f"hash values before: {hash_vals_before}")
    print(f"interpolated values: {hash_vals}")

def test_forward():
    args = Args()
    grid = Grid(args, layer=0)

    X = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).to(args.device)
    hash_vals = grid(X)
    
    print(f"hash values: {hash_vals}")


if __name__ == '__main__':
    # test_getCubeIdxs()
    # test_hashFunction()
    test_linearInterpolation()
    # test_forward()