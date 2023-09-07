import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from args import Args

class Plotter():
    def __init__(self, args:Args) -> None:
        self.args = args

    def cameraPoses(self, rays):
        """
        Visualize camera positions and viewing directions.
        Returns:
            rays: rays; dict of numpy array of shape (N, ro+rd, H, W, 3)
        """
        # Extract positions and viewing directions from the input numpy array
        H = rays["test"].shape[2]
        W = rays["test"].shape[3]
        positions = rays["test"][:,0,H//2,W//2,:]
        view_directions = rays["test"][:,1,H//2,W//2,:]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a color map and normalize it based on the number of camera poses
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=positions.shape[0])
        sm = ScalarMappable(cmap=cmap, norm=norm)

        # calculate step size
        step = positions.shape[0] // self.args.nb_camera_poses
        if step == 0:
            step = 1

        # Plot camera positions as dots with color gradient
        for i in range(0, len(positions), step):
            color = cmap(norm(i))
            ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], c=color, marker='o', label='Camera Positions')

        # Plot camera viewing directions as lines with the same color gradient
        for i in range(0, len(view_directions), step):
            start = positions[i]
            end = positions[i] + view_directions[i]
            color = cmap(norm(i))
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c=color, label='Viewing Directions')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Poses Visualization')
        
        # Create a color bar to indicate the gradient
        sm.set_array(np.linspace(0, positions.shape[0], step))
        cbar = plt.colorbar(sm, ax=ax, label='Pose Index')
        
        #ax.legend()
        
        plt.show()