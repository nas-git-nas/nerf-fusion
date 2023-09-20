import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from args import Args

class Plotter():
    def __init__(self, args:Args) -> None:
        self.args = args

        self.fig = None
        self.ax = None
        self.colourMap = None

    def showTestImgs(self, images_list1, images_list2, losses=None):
        """
        Plot two lists of images side by side in two columns.

        Parameters:
        - images_list1: List of numpy arrays representing the first ground truth.
        - images_list2: List of numpy arrays representing the estimation.
        - losses: List of losses for each image.
        """

        num_images1 = len(images_list1)
        num_images2 = len(images_list2)

        if num_images1 != num_images2:
            raise ValueError("Both lists of images must have the same number of elements.")

        fig, axes = plt.subplots(num_images1, 2, figsize=(10, 5 * num_images1))

        for i in range(num_images1):
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]

            image1 = images_list1[i]
            image2 = images_list2[i]

            title1 = "GT: image " + str(i)
            title2 = "Est: image " + str(i)

            if losses is not None:
                title2 += ", loss: " + str(np.round(losses[i],4))

            ax1.set_title(title1)
            ax2.set_title(title2)

            ax1.imshow(image1)
            ax1.axis('off')

            ax2.imshow(image2)
            ax2.axis('off')

        plt.tight_layout()
        plt.show()

    def showPlot(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Camera Poses Visualization')
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        
        #ax.legend()
        plt.show()

    def cameraPositions(self, rays):
        """
        Visualize camera positions and viewing directions.
        Returns:
            rays: rays; numpy array of shape (N, ro+rd, H, W, 3)
        """
        # Extract positions and viewing directions from the input numpy array
        H = rays.shape[2]
        W = rays.shape[3]
        positions = rays[:,0,H//2,W//2,:]

        # create figure if not already done
        self._createFigure()

        # Create a color map and normalize it based on the number of camera poses
        self._createColourMap(nb_steps=positions.shape[0])

        # calculate step size
        step_I, _, _ = self._representationStep(nb_images=positions.shape[0])

        # Plot camera positions as dots with color gradient
        for i in range(0, len(positions), step_I):
            color = self.colourMap["cmap"](self.colourMap["norm"](i))
            self.ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], c=color, marker='o', label='Camera Positions')

        # # Plot camera viewing directions as lines with the same color gradient
        # view_directions = rays["test"][:,1,H//2,W//2,:]
        # for i in range(0, len(view_directions), step_I):
        #     start = positions[i]
        #     end = positions[i] + view_directions[i]
        #     color = cmap(norm(i))
        #     self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c=color, label='Viewing Directions')

    def cameraFoV(self, rays):
        """
        Visualize camera positions and viewing directions.
        Returns:
            rays: rays; numpy array of shape (N, ro+rd, H, W, 3)
        """
        # Extract positions and viewing directions from the input numpy array
        H = rays.shape[2]
        W = rays.shape[3]
        positions = rays[:,0,H//2,W//2,:]
        view_directions = rays[:,1,H//2,W//2,:]

        # create figure if not already done
        self._createFigure()

        # Create a color map and normalize it based on the number of camera poses
        self._createColourMap(nb_steps=positions.shape[0])

        # calculate step size
        step_I, _, _ = self._representationStep(nb_images=positions.shape[0])

        # plot field of view of camera
        all_corners = np.array([rays[:,0,0,0,:], rays[:,0,H-1,0,:], 
                                rays[:,0,H-1,W-1,:], rays[:,0,0,W-1,:]]) # (4, N, 3)
        all_corners = all_corners.transpose(1, 0, 2) # (N, 4, 3)
        all_rays = np.array([rays[:,1,0,0,:], rays[:,1,H-1,0,:], 
                             rays[:,1,H-1,W-1,:], rays[:,1,0,W-1,:]]) # (4, N, 3)
        all_rays = all_rays.transpose(1, 0, 2) # (N, 4, 3)
        for i in range(0, all_corners.shape[0], step_I):
            fov_polygon = self._createFoVpolygon(all_corners[i], all_rays[i])
            color = self.colourMap["cmap"](self.colourMap["norm"](i))
            self.ax.add_collection3d(Poly3DCollection(fov_polygon, alpha=0.18, edgecolor=color, facecolor=color))

    def cameraRays(self, positions, directions, rays, colours_gt):
        """
        Visualize sampled rays.
        Args:
            positions: sampled positions; np.array (I*R*M, 3)
            directions: sampled directions; np.array (I*R, 3)
            rays: rays; numpy array of shape (I, ro+rd, H, W, 3)
            colours_gt: colours; numpy array of shape (I*R, 4)
        """
        positions = positions.reshape(self.args.I, self.args.R, self.args.M, 3)
        directions = directions.reshape(self.args.I, self.args.R, 3)
        colours_gt = colours_gt.reshape(self.args.I, self.args.R, 4)
        H = rays.shape[2]
        W = rays.shape[3]
        c_positions = rays[:,0,H//2,W//2,:] # (I, 3)

        # create figure if not already done
        self._createFigure()

        # Create a color map and normalize it based on the number of camera poses
        self._createColourMap(nb_steps=positions.shape[0])

        # calculate step size
        step_I, step_R, step_M = self._representationStep(nb_images=positions.shape[0])

        # calculate max. step that one can take on ray while keeping inside of [-1,1]**3
        ray_origins = np.repeat(c_positions, self.args.R, axis=0) # (I*R, 3)
        ray_directions = directions.reshape(-1, 3) # (I*R, 3)
        ray_sign = np.sign(ray_directions) # (I*R, 3), mirror problem if direction is negative
        t_max = np.min( (1 - ray_sign*ray_origins) / (ray_sign*ray_directions), axis=1) # (I*R,)
        t_max = t_max.reshape(self.args.I, self.args.R) # (I, R)

        # Plot camera positions as dots with color gradient
        for i in range(0, positions.shape[0], step_I):
            # determine colour
            color = self.colourMap["cmap"](self.colourMap["norm"](i))

            for j in range(0, positions.shape[1], step_R):
                # determine length of ray
                # direction_scale = np.linalg.norm( positions[i,j,:,:] - c_positions[i].reshape(1,3), axis=1 )
                # direction_scale = np.max(direction_scale)
                direction_scale = t_max[i,j]

                # plot all sample rays
                start = c_positions[i]
                end = c_positions[i] + direction_scale * directions[i,j] / np.linalg.norm(directions[i,j])
                self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c=color, label='Ray Direction')

                # plot all sample points
                for k in range(0, positions.shape[2], step_M):
                    self.ax.scatter(positions[i,j,k,0], positions[i,j,k,1], positions[i,j,k,2], c=color, marker='o', label='Point Positions')  
    
    def _createFigure(self):
        if self.fig is not None:
            return
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def _createColourMap(self, nb_steps):
        if self.colourMap is not None:
            return

        # Create a color map and normalize it based on the number of camera poses
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=nb_steps)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        self.colourMap = {"cmap": cmap, "norm": norm, "sm": sm}

        # Create a color bar to indicate the gradient
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=self.ax, label='Pose Index')

    def _representationStep(self, nb_images):
        # calculate step size
        step_I = nb_images // self.args.plot_nb_I
        if step_I == 0:
            step_I = 1

        step_R = self.args.R // self.args.plot_nb_R
        if step_R == 0:
            step_R = 1

        step_M = self.args.M // self.args.plot_nb_M
        if step_M == 0:
            step_M = 1

        return step_I, step_R, step_M
    
    def _createFoVpolygon(self, corners, rays):
        # scale rays
        rays = 0.3 * ( rays.T / np.linalg.norm(rays, axis=1) ).T

        # Calculate field of view vertices
        fov_vertices1 = np.zeros((5, 3))
        fov_vertices1[0, :] = corners[0, :]  # Start from the first corner
        for j in range(1, 4):
            fov_vertices1[j, :] = corners[j, :]
        fov_vertices1[4, :] = corners[0, :]  # Close the polygon by going back to the first corner

        fov_vertices2 = np.zeros((5, 3))
        fov_vertices2[0, :] = corners[0, :]
        fov_vertices2[1, :] = corners[0, :] + rays[0, :] 
        fov_vertices2[2, :] = corners[1, :] + rays[1, :]
        fov_vertices2[3, :] = corners[1, :]
        fov_vertices2[4, :] = corners[0, :]

        fov_vertices3 = np.zeros((5, 3))
        fov_vertices3[0, :] = corners[1, :]
        fov_vertices3[1, :] = corners[1, :] + rays[1, :]
        fov_vertices3[2, :] = corners[2, :] + rays[2, :]
        fov_vertices3[3, :] = corners[2, :]
        fov_vertices3[4, :] = corners[1, :]

        fov_vertices4 = np.zeros((5, 3))
        fov_vertices4[0, :] = corners[2, :]
        fov_vertices4[1, :] = corners[2, :] + rays[2, :]
        fov_vertices4[2, :] = corners[3, :] + rays[3, :]
        fov_vertices4[3, :] = corners[3, :]
        fov_vertices4[4, :] = corners[2, :]

        fov_vertices5 = np.zeros((5, 3))
        fov_vertices5[0, :] = corners[3, :]
        fov_vertices5[1, :] = corners[3, :] + rays[3, :]
        fov_vertices5[2, :] = corners[0, :] + rays[0, :]
        fov_vertices5[3, :] = corners[0, :]
        fov_vertices5[4, :] = corners[3, :]

        fov_vertices6 = np.zeros((5, 3))
        fov_vertices6[0, :] = corners[0, :] + rays[0, :] 
        for j in range(1, 4):
            fov_vertices6[j, :] = corners[j, :] + rays[j, :] 
        fov_vertices6[4, :] = corners[0, :] + rays[0, :] 

        # Create a polygon to represent the field of view (transparent)
        fov_polygon = [fov_vertices1, fov_vertices2, fov_vertices3, fov_vertices4, fov_vertices5, fov_vertices6]

        return fov_polygon
    


        
