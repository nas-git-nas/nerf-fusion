import numpy as np
import torch

from torch import nn

from args import Args


class Grid(nn.Module):
    def __init__(self, args:Args, layer) -> None:

        self.args = args
        
        # calc. growth factor
        growth_factor = np.exp( (np.log(args.res_max) - np.log(args.res_min)) / (args.L - 1))
        self.res = (growth_factor**layer) * args.res_min

        # create learnable gird
        self.hash_table = nn.Parameter(torch.zeros(self.args.T, self.args.F))

        # prime numbers for hash function
        self.primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.int64)
        
    def forward(self, X):
        """
        Forward pass of the grid
        Args:
            X: batch of points, torch.tensor (N, D)
        Returns:
            hash_vals: hashed and linear interpolated points, torch.tensor (N, F)
        """
        # get hash table indices
        with torch.no_grad():
            cube_idxs = self._getCubeIdxs(X, self.args)
            hash_idxs = self._hashFunction(cube_idxs, self.args)

        # get hash table values (N, 2**D, F)
        hash_vals = self.hash_table[hash_idxs]

        # 3-linear interpolation (N, F)
        hash_vals = self._linearInterpolation(X, hash_vals)

        return hash_vals

    def _getCubeIdxs(self, X, args:Args):
        """
        Get cube indices for each point in X
        Args:
            X: torch.tensor of shape (N, D)
            args: Args object
        Returns:
            cube_indices: torch.tensor of shape (N, 2**D, D)
        """
        # get grid indices
        X_scaled = X * (self.res-1)
        X_floor = torch.floor(X_scaled)
        X_ceil = torch.ceil(X_scaled)

        # get cube indices
        cube_idxs = torch.empty(X.shape[0], np.power(2, args.nb_dim), args.nb_dim)
        cube_idxs[:, 0, :] = X_floor
        cube_idxs[:, 1, :] = torch.hstack(X_floor[:,0].reshape(-1,1), X_floor[:,1].reshape(-1,1), X_ceil[:,2].reshape(-1,1))
        cube_idxs[:, 2, :] = torch.hstack(X_floor[:,0].reshape(-1,1), X_ceil[:,1].reshape(-1,1), X_floor[:,2].reshape(-1,1))
        cube_idxs[:, 3, :] = torch.hstack(X_floor[:,0].reshape(-1,1), X_ceil[:,1].reshape(-1,1), X_ceil[:,2].reshape(-1,1))
        cube_idxs[:, 4, :] = torch.hstack(X_ceil[:,0].reshape(-1,1), X_floor[:,1].reshape(-1,1), X_floor[:,2].reshape(-1,1))
        cube_idxs[:, 5, :] = torch.hstack(X_ceil[:,0].reshape(-1,1), X_floor[:,1].reshape(-1,1), X_ceil[:,2].reshape(-1,1))
        cube_idxs[:, 6, :] = torch.hstack(X_ceil[:,0].reshape(-1,1), X_ceil[:,1].reshape(-1,1), X_floor[:,2].reshape(-1,1))
        cube_idxs[:, 7, :] = X_ceil

        return cube_idxs
    
    def _hashFunction(self, cube_idxs, args:Args):
        """
        Map hash cube indices to hash table indices
        Args:
            cube_indices: cube indices, tensor (N, 2**D, D)
            args: arguments, Args object
        Returns:
            hash_idxs: hash table indices, tensor (N, 2**D)
        """
        # do hash mapping
        prime_table = self.primes.reshape(1,1,-1).repeat(cube_idxs.shape[0], cube_idxs.shape[1], 1)
        hash_map = cube_idxs * prime_table

        hash_idxs = hash_map[:,:,0]
        for i in range(1, args.nb_dim):
            hash_idxs = torch.bitwise_xor(hash_idxs, hash_map[:,:,i])
        hash_idxs = torch.remainder(hash_idxs, args.hash_table_size[0])

        return hash_idxs
    
    def _linearInterpolation(self, X, hash_vals):
        """
        D-linear interpolation of hash values
        Args:
            X: batch of points, tensor (N, D)
            hash_vals: hash values, tensor (N, 2**D, F)
        Returns:
            hash_vals: interpolated hash values, tensor (N, F)
        """
        # get interpolation weights
        X_scaled = X * (self.res-1)
        X_floor = torch.floor(X_scaled)
        weights = X_scaled - X_floor

        # interpolate along x
        p00 = hash_vals[:,0,:] * (1-weights[:,0]).reshape(-1,1) + hash_vals[:,4,:] * weights[:,0].reshape(-1,1)
        p01 = hash_vals[:,1,:] * (1-weights[:,0]).reshape(-1,1) + hash_vals[:,5,:] * weights[:,0].reshape(-1,1)
        p10 = hash_vals[:,2,:] * (1-weights[:,0]).reshape(-1,1) + hash_vals[:,6,:] * weights[:,0].reshape(-1,1)
        p11 = hash_vals[:,3,:] * (1-weights[:,0]).reshape(-1,1) + hash_vals[:,7,:] * weights[:,0].reshape(-1,1)

        # interpolate along y
        p0 = p00 * (1-weights[:,1]).reshape(-1,1) + p10 * weights[:,1].reshape(-1,1)
        p1 = p01 * (1-weights[:,1]).reshape(-1,1) + p11 * weights[:,1].reshape(-1,1)

        # interpolate along z
        hash_vals = p0 * (1-weights[:,2]).reshape(-1,1) + p1 * weights[:,2].reshape(-1,1)

        return hash_vals
