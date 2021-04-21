"""3D visualization tools."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from enzynet.PDB import PDB_backbone
from enzynet.volume import adjust_size, coords_to_volume, coords_center_to_zero, weights_to_volume


def visualize_pdb(pdb_id, p=5, v_size=32, num=1, weights=None,
                  max_radius=40, noise_treatment=True):
    """Plots PDB in a volume and saves it in a file."""
    # Get coordinates.
    pdb = PDB_backbone(pdb_id)
    pdb.get_coords_extended(p=p)

    if weights != None:
        pdb.get_weights_extended(p=p, weights=weights)

    # Center to 0.
    coords = coords_center_to_zero(pdb.backbone_coords_ext)

    # Adjust size.
    coords = adjust_size(coords, v_size, max_radius)

    if weights == None:
        # Convert to volume.
        volume = coords_to_volume(coords, v_size, noise_treatment=noise_treatment)
    else:
        # Converts to volume of weights.
        volume = weights_to_volume(coords, pdb.backbone_weights_ext, v_size,
                                   noise_treatment=noise_treatment)
    # Plot.
    plot_volume(volume, pdb_id, v_size, num, weights=weights)


# 3D plot, sources: http://stackoverflow.com/a/35978146/4124317
#                   https://dawes.wordpress.com/2014/06/27/publication-ready-3d-figures-from-matplotlib/
def plot_volume(volume, pdb_id, v_size, num, weights=None):
    """Plots volume in 3D, interpreting the coordinates as voxels."""
    # Initialization.
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(4,4))
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    # Parameters.
    len_vol = volume.shape[0]

    # Set position of the view.
    ax.view_init(elev=20, azim=135)

    # Hide tick labels.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Plot.
    if weights == None:
        plot_matrix(ax, volume)
    else:
        plot_matrix_of_weights(ax, volume)

    # Tick at every unit.
    ax.set_xticks(np.arange(len_vol))
    ax.set_yticks(np.arange(len_vol))
    ax.set_zticks(np.arange(len_vol))

    # Min and max that can be seen.
    ax.set_xlim(0, len_vol-1)
    ax.set_ylim(0, len_vol-1)
    ax.set_zlim(0, len_vol-1)

    # Clear grid.
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Change thickness of grid.
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.1

    # Change thickness of ticks.
    ax.xaxis._axinfo["tick"]['linewidth'] = 0.1
    ax.yaxis._axinfo["tick"]['linewidth'] = 0.1
    ax.zaxis._axinfo["tick"]['linewidth'] = 0.1

    # Change tick placement.
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.2
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.2
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.2
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.2

    # Save.
    plt.savefig('../scripts/volume/' + str(pdb_id) + '_' + str(v_size) + '_' +
                str(weights) + '_' + str(num) + '.pdf')


def cuboid_data(pos, size=(1,1,1)):
    """Gets coordinates of cuboid."""
    # Gets the (left, outside, bottom) point.
    o = [a - b / 2 for a, b in zip(pos, size)]

    # Get the length, width, and height.
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]] for i in range(4)]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]

    return x, y, z


def plot_cube_at(pos=(0,0,0), ax=None):
    """Plots a cube element at position pos."""
    if ax != None:
        X, Y, Z = cuboid_data(pos)
        ax.plot_surface(X, Y, Z, color='g', rstride=1, cstride=1, alpha=1)


def plot_cube_weights_at(pos=(0,0,0), ax=None, color='g'):
    """Plots a cube element at position pos."""
    if ax != None:
        X, Y, Z = cuboid_data(pos)
        ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=1)


def plot_matrix(ax, matrix):
    """Plots cubes from a volumic matrix."""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i,j,k] == 1:
                    plot_cube_at(pos=(i-0.5,j-0.5,k-0.5), ax=ax)


def plot_matrix_of_weights(ax, matrix_of_weights):
    """Plots cubes from a volumic matrix."""
    # Initialization.
    min_value = np.amin(matrix_of_weights)
    max_value = np.amax(matrix_of_weights)
    n_colors = 101

    # Check if matrix of weights or not.
    if min_value == max_value == 1:
        return plot_matrix(ax, matrix_of_weights)

    # Generate colors.
    cm = plt.get_cmap('seismic')
    cgen = [cm(1.*i/n_colors) for i in range(n_colors)]

    # Plot cubes.
    for i in range(matrix_of_weights.shape[0]):
        for j in range(matrix_of_weights.shape[1]):
            for k in range(matrix_of_weights.shape[2]):
                if matrix_of_weights[i,j,k] != 0:
                    # Translate to [0,100].
                    normalized_weight = (matrix_of_weights[i,j,k] - min_value)/ \
                                        (max_value - min_value)
                    normalized_weight = int(100*normalized_weight)

                    # Plot cube with color.
                    plot_cube_weights_at(pos=(i-0.5,j-0.5,k-0.5), ax=ax,
                                         color=cgen[normalized_weight])


if __name__ == '__main__':
    visualize_pdb('2Q3Z', p=0, v_size=32, weights=None)
