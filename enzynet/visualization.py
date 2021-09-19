"""3D visualization tools."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from typing import Optional, Text, Tuple

from absl import app
from absl import flags

import os

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from enzynet import constants
from enzynet import pdb
from enzynet import volume

FLAGS = flags.FLAGS

flags.DEFINE_string('pdb_id', default='2Q3Z', help='PDB ID to be visualized.')
flags.DEFINE_integer('v_size', lower_bound=1, default=32, help='Size of each '
                     'dimension of the grid where enzymes are represented. '
                     'Figure 2 of the paper shows 2Q3Z results for v_size '
                     'values of 32, 64 and 96.')
flags.DEFINE_integer('p', lower_bound=0, default=0, help='Number of '
                     'interpolated points between two consecutive represented '
                     'atoms. This parameter is used for finer grid '
                     'representations in order to draw lines between '
                     'consecutive points.')
flags.DEFINE_enum('weight_type', default=None,
                  enum_values=['charge', 'hydropathy', 'isoelectric'],
                  help='If None, binary voxels of the atoms are shown. '
                  'Otherwise, displays visualization of the corresponding '
                  'weight type.')


def visualize_pdb(pdb_id: Text, p: int = 5, v_size: int = 32, num: int = 1,
                  weight_type: Optional[Text] = None, max_radius: int = 40,
                  noise_treatment: bool = True) -> None:
    """Plots PDB in a volume and saves it in a file."""
    # Get coordinates.
    pdb_backbone = pdb.PDBBackbone(pdb_id)
    pdb_backbone.get_coords_extended(p=p)

    if weight_type is not None:
        pdb_backbone.get_weights_extended(p=p, weight_type=weight_type)

    # Center to 0.
    coords = volume.coords_center_to_zero(pdb_backbone.backbone_coords_ext)

    # Adjust size.
    coords = volume.adjust_size(coords, v_size, max_radius)

    if weight_type is None:
        # Convert to volume.
        vol = volume.coords_to_volume(coords, v_size,
                                      noise_treatment=noise_treatment)
    else:
        # Converts to volume of weights.
        vol = volume.weights_to_volume(
            coords, pdb_backbone.backbone_weights_ext, v_size,
            noise_treatment=noise_treatment)
    # Plot.
    plot_volume(vol, pdb_id, v_size, num, weight_type=weight_type)


# 3D plot, sources: http://stackoverflow.com/a/35978146/4124317
#                   https://dawes.wordpress.com/2014/06/27/publication-ready-3d-figures-from-matplotlib/
def plot_volume(vol: np.ndarray, pdb_id: Text, v_size: int, num: int,
                weight_type: Optional[Text] = None):
    """Plots volume in 3D, interpreting the coordinates as voxels."""
    # Initialization.
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(projection='3d')

    # Reproduces the functionality of ax.set_aspect('equal').
    # Source: https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    # Parameters.
    len_vol = vol.shape[0]

    # Set position of the view.
    ax.view_init(elev=20, azim=135)

    # Hide tick labels.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Plot.
    if weight_type is None:
        plot_matrix(ax, vol)
    else:
        plot_matrix_of_weights(ax, vol)

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
    ax.xaxis._axinfo["tick"]['linewidth'][True] = 0.1
    ax.yaxis._axinfo["tick"]['linewidth'][True] = 0.1
    ax.zaxis._axinfo["tick"]['linewidth'][True] = 0.1

    # Change tick placement.
    for factor_type, val in zip(['inward_factor', 'outward_factor'], [0, 0.2]):
        ax.xaxis._axinfo['tick'][factor_type] = val
        ax.yaxis._axinfo['tick'][factor_type] = val
        ax.zaxis._axinfo['tick'][factor_type] = val

    # Save.
    plt.savefig(os.path.join(constants.VISUALIZATION_DIR,
                             f'{pdb_id}_{v_size}_{weight_type}_{num}.pdf'))


def cuboid_data(
        pos: Tuple[float, float, float],
        size: Tuple[int, int, int]=(1,1,1)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return np.array(x), np.array(y), np.array(z)


def plot_cube_at(pos: Tuple[float, float, float] = (0, 0, 0),
                 ax: Optional[plt.gca] = None,
                 color: Text = 'g') -> None:
    """Plots a cube element at position pos."""
    lightsource = mcolors.LightSource(azdeg=135, altdeg=0)
    if ax != None:
        X, Y, Z = cuboid_data(pos)
        ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=1,
                        lightsource=lightsource)


def plot_matrix(ax: Optional[plt.gca], matrix: np.ndarray) -> None:
    """Plots cubes from a volumic matrix."""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i, j, k] == 1:
                    plot_cube_at(pos=(i-0.5, j-0.5, k-0.5), ax=ax)


def plot_matrix_of_weights(ax: plt.gca, matrix_of_weights: np.ndarray) -> None:
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
                    normalized_weight = (matrix_of_weights[i, j, k] - min_value)/ \
                                        (max_value - min_value)
                    normalized_weight = int(100*normalized_weight)

                    # Plot cube with color.
                    plot_cube_at(pos=(i-0.5, j-0.5, k-0.5), ax=ax,
                                 color=cgen[normalized_weight])


def main(_):
    visualize_pdb(FLAGS.pdb_id, p=FLAGS.p, v_size=FLAGS.v_size,
                  weight_type=FLAGS.weight_type)


if __name__ == '__main__':
    app.run(main)
