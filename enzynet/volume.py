'Volume generation and augmentation'

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import os.path

import numpy as np

from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

from enzynet import PDB_backbone
from enzynet import threadsafe_generator


current_directory = os.path.dirname(os.path.abspath(__file__))
precomputed_path = os.path.join(current_directory, '../files/precomputed/')
PDB_path = os.path.join(current_directory, '../files/PDB/')

class VolumeDataGenerator(object):
    """

    Generates batches of volumes containing 3D structure of enzymes as well
    as their associated class labels on the fly.

    To be passed as argument in the fit_generator function of Keras.


    Parameters
    ----------
    v_size : int (optional, default is 32)
        Size of each side of the output volumes.

    flips : tuple of floats (optional, default is (0.2, 0.2, 0.2))
        Probabilities that the volumes are flipped respectively with respect
        to x, y, and z.

    batch_size : int (optional, default is 32)
        Number of samples in output array of each iteration of the 'generate'
        method.

    directory_precomputed : string (optional, default points to 'files/precomputed')
        Path of the precomputed files.

    directory_pdb : string (optional, default points to 'files/PDB')
        Path of the PDB files.

    shuffle : boolean (optional, default is True)
        If True, shuffles order of exploration.

    p : int (optional, default is 5)
        Interpolation of enzymes with p added coordinates between each pair
        of consecutive atoms.

    max_radius : float (optional, default is 40)
        Maximum radius of sphere that will completely fit into the volume.

    noise_treatment : boolean (optional, default is False)
        If True, voxels with no direct neighbor will be deleted.

    weights : list of strings (optional, default is [])
        List of weights (among the values ['hydropathy', 'charge']) to consider
        as additional channels.

    scaling_weights : boolean (optional, default is True)
        If True, divides all weights by the weight that is maximum in absolute
        value.


    Example
    -------
    >>> from enzynet import VolumeDataGenerator
    >>> from enzynet import read_dict

    >>> dictionary = read_dict('../datasets/dataset_single.csv')
    >>> partition_red = read_dict('../../datasets/partition_single_red.csv')
    >>> exec("partition_red['train'] = " + partition_red['train'])

    >>> generator = VolumeDataGenerator(v_size = 64, flips = (0.2, 0.2, 0.2),
                                        batch_size = 32, shuffle = True, p = 5,
                                        max_radius = 40, noise_treatment = False,
                                        weights = [], scaling_weights = True)
    >>> generator = generator.generate(dictionary, partition_red['train'])

    """
    def __init__(self, v_size = 32, flips = (0.2, 0.2, 0.2), batch_size = 32,
                 directory_precomputed = precomputed_path, directory_pdb = PDB_path,
                 shuffle = True, p = 5, max_radius = 40, noise_treatment = False,
                 weights = [], scaling_weights = True):
        'Initialization'
        self.v_size = v_size
        self.flips = flips
        self.batch_size = batch_size
        self.directory_precomputed = directory_precomputed
        self.directory_pdb = directory_pdb
        self.shuffle = shuffle
        self.p = p
        self.max_radius = max_radius
        self.noise_treatment = noise_treatment
        self.weights = weights
        self.n_channels = max(1, len(weights))
        self.scaling_weights = scaling_weights

    def check_precomputed(self, dictionary):
        'Checks if all coordinates and weights have been precomputed, and precomputes them otherwise'
        # Initialization
        list_enzymes = list(dictionary)
        counter = 0

        # Loop over all enzymes
        for pdb_id in tqdm(list_enzymes):
            # Find name of paths
            names = [precomputed_name(pdb_id, self.directory_precomputed, 'coords', self.p)] + \
                    [precomputed_name(pdb_id, self.directory_precomputed, 'weights', self.p,
                                      weight, self.scaling_weights)
                     for weight in self.weights]

            # Precomputes all files
            if all([os.path.isfile(name) for name in names]): # Check if all already exist
                pass
            else: # Precomputes files otherwise
                save_coords_weights(pdb_id, self.weights, self.p, self.scaling_weights,
                                    self.directory_pdb, self.directory_precomputed)
                counter += 1
        print("Had to compute files of {0} enzymes.".format(counter))

    @threadsafe_generator
    def generate(self, dictionary, list_enzymes):
        'Generates batches of enzymes from list_enzymes'
        # Infinite loop
        while 1:
            # Generate order of exploration of augmented dataset
            indexes = \
                self.__get_exploration_order(list_enzymes)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find local list of enzymes
                list_enzymes_temp = [list_enzymes[k]
                                     for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate batch by knowing enzymes and their rotations
                X, y = self.__data_augmentation(dictionary, list_enzymes_temp)

                yield X, y

    def __get_exploration_order(self, list_enzymes):
        'Generates indexes of exploration'
        # Find exploration order
        indexes = np.arange(len(list_enzymes))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_augmentation(self, dictionary, list_enzymes):
        'Returns augmented data with batch_size enzymes' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((len(list_enzymes), # n_enzymes
                      self.v_size, # dimension w.r.t. x
                      self.v_size, # dimension w.r.t. y
                      self.v_size, # dimension w.r.t. z
                      self.n_channels)) # n_channels
        y = np.empty((len(list_enzymes)), dtype = int)

        # Computations
        for i in range(len(list_enzymes)):
            # Store class
            y[i] = dictionary[list_enzymes[i]]

            # Load precomputed coordinates
            coords = load_coords(list_enzymes[i], self.p, self.directory_precomputed)
            coords = coords_center_to_zero(coords) # Center coords on zero
            coords = adjust_size(coords, vsize = self.v_size,
                                 max_radius = self.max_radius) # Adjust size

            # Get weights
            local_weights = []
            for weight in self.weights:
                local_weight = load_weights(list_enzymes[i], weight, self.p,
                                            self.scaling_weights, self.directory_precomputed) # Compute extended weights
                local_weights += [local_weight] # Store

            # PCA
            coords = PCA(n_components = 3).fit_transform(coords)

            # Do flip
            coords_temp = flip_around_axis(coords,
                                           axis = self.flips)

            if len(self.weights) == 0:
                # Convert to volume and store
                X[i, :, :, :, 0] = \
                    coords_to_volume(coords_temp, self.v_size,
                                     noise_treatment = self.noise_treatment)

            else:
                # Compute to weights of volume and store
                for k in range(0, self.n_channels):
                    X[i, :, :, :, k] = \
                        weights_to_volume(coords_temp, local_weights[k], self.v_size,
                                          noise_treatment = self.noise_treatment)[1]

        return X, sparsify(y)


def coords_to_volume(coords, v_size, noise_treatment = False):
    'Converts coordinates to binary voxels' # Input is centered on [0,0,0]
    # Initialization
    volume = np.zeros((v_size, v_size, v_size))
    coords = coords + np.full((coords.shape[0], 3), (v_size-1)/2) # Translate center
    coords = coords.astype(int) # Round components
    coords = unique_rows(coords) # Delete redundant rows

    # Computations
    for i in range(coords.shape[0]):
        if all(valeur < v_size for valeur in coords[i,:]) and \
           all(valeur >= 0 for valeur in coords[i,:]):
           volume[coords[i,0], coords[i,1], coords[i,2]] = 1

    if noise_treatment == True: # Remove noise
        volume = remove_noise(coords, volume)

    return volume

def weights_to_volume(coords, weights, v_size, noise_treatment = False):
    'Converts coordinates to voxels with weights' # Input is centered on [0,0,0]
    # Initialization
    volume_coords = coords_to_volume(coords, v_size, noise_treatment = noise_treatment)
    volume_weights = np.zeros((v_size, v_size, v_size))
    coords = coords + np.full((coords.shape[0], 3), (v_size-1)/2) # Translate center
    coords = coords.astype(int) # Round components

    # Computations
    for i in range(coords.shape[0]):
        if all(valeur < v_size for valeur in coords[i,:]) and \
           all(valeur >= 0 for valeur in coords[i,:]):
           if volume_coords[coords[i,0], coords[i,1], coords[i,2]] == 1:
               volume_weights[coords[i,0], coords[i,1], coords[i,2]] = weights[i]

    return volume_coords, volume_weights

def coords_center_to_zero(coords):
    'Centering coordinates on [0,0,0]'
    barycenter = get_barycenter(coords)
    return coords - np.full((coords.shape[0], 3), barycenter)

def adjust_size(coords, vsize = 32, max_radius = 40):
    return np.multiply((vsize/2-1)/max_radius, coords)

def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 6
    return np.array([[1 if y[i] == j+1 else 0 for j in range(n_classes)]
                      for i in range(y.shape[0])])

def random_rotation(coords):
    'Performs a random rotation' # Input is centered on [0,0,0]
    M = rotation_matrix(random = True)
    return np.dot(coords, M)

def rotation_around_axis(coords, factor = 0, axis = [1,0,0]):
    'Rotation of coords around axis'
    # Adapted from stackoverflow.com/a/6802723
    theta = factor * 2 * np.pi
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    M = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                  [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                  [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return np.dot(coords, M)

def flip_around_axis(coords, axis = (0.2, 0.2, 0.2)):
    'Flips coordinates randomly w.r.t. each axis with its associated probability'
    for col in range(3):
        if np.random.binomial(1, axis[col]):
            coords[:,col] = np.negative(coords[:,col])
    return coords

def characterized_rotation(coords, theta = 0, phi = 0, z = 0):
    'Performs a rotation given theta, phi and z in [0,1]' # Input is centered on [0,0,0]
    M = rotation_matrix(theta = theta, phi = phi, z = z)
    return np.dot(coords, M)

def rotation_matrix(random = False, theta = 0, phi = 0, z = 0):
    'Creates a rotation matrix'
    # Adapted from: http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    # Initialization
    if random == True:
        randnums = np.random.uniform(size=(3,))
        theta, phi, z = randnums
    theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0  # For magnitude of pole deflection.
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)

    return M

def get_barycenter(coords):
    'Gets barycenter point of a Nx3 matrix'
    return np.array([np.mean(coords, axis = 0)])

def get_barycenter_and_radius(coords):
    'Gets barycenter and radius of the sphere including all of the points'
    barycenter = get_barycenter(coords)
    tree = KDTree(coords)
    dist, ind = tree.query(barycenter, k = coords.shape[0])
    return barycenter, dist[0][-1]

def get_barycenter_and_distances(coords):
    'Gets barycenter and distances of all points from it'
    barycenter = get_barycenter(coords)
    tree = KDTree(coords)
    dist, ind = tree.query(barycenter, k = coords.shape[0])
    return barycenter, dist[0]

def unique_rows(a):
    'Deletes redundant rows from an array'
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def remove_noise(coords, volume):
    'Removes isolated atoms from voxel structure'
    # Parameters
    v_size = volume.shape[0]

    # Computations
    for i in range(coords.shape[0]):
        if all(valeur < v_size-1 for valeur in coords[i,:]) and \
           all(valeur > 0 for valeur in coords[i,:]): # Point inside volume
            if np.sum(volume[coords[i,0]-1:coords[i,0]+2,
                             coords[i,1]-1:coords[i,1]+2,
                             coords[i,2]-1:coords[i,2]+2]) == 1: # Isolated point
                volume[coords[i,0]-1:coords[i,0]+2,
                       coords[i,1]-1:coords[i,1]+2,
                       coords[i,2]-1:coords[i,2]+2] = np.zeros((3,3,3))

    return volume

def precomputed_name(pdb_id, path, type_file, desired_p, weights_name = None, scaling = True):
    'Returns path in string of precomputed file'
    if type_file == 'coords':
        return os.path.join(path, pdb_id.lower() + '_coords_p' + str(desired_p) + '.npy')
    elif type_file == 'weights':
        return os.path.join(path, pdb_id.lower() + '_' + weights_name + '_p' + str(desired_p) + '_scaling' + str(scaling) + '.npy')

def save_coords_weights(pdb_id, list_weights, desired_p, scaling_weights,
                        source_path, dest_path):
    'Computes coordinates and weights and saves them into .npy files'
    # Initialize local PDB
    local_PDB = PDB_backbone(pdb_id = pdb_id, path = source_path)

    # Coordinates
    local_PDB.get_coords_extended(p = desired_p) # Compute
    coords = local_PDB.backbone_coords_ext # Store
    np.save(precomputed_name(pdb_id, dest_path, 'coords', desired_p), coords) # Save

    # Weights
    for weights_name in list_weights:
        local_PDB.get_weights_extended(desired_p, weights = weights_name,
                                       scaling = scaling_weights) # Compute
        weights = local_PDB.backbone_weights_ext # Store
        np.save(precomputed_name(pdb_id, dest_path, 'weights', desired_p, weights_name, scaling_weights),
                weights) # Save

def load_coords(pdb_id, desired_p, source_path):
    'Loads precomputed coordinates'
    return np.load(precomputed_name(pdb_id, source_path, 'coords', desired_p))

def load_weights(pdb_id, weights_name, desired_p, scaling, source_path):
    'Loads precomputed weights'
    return np.load(precomputed_name(pdb_id, source_path, 'weights', desired_p, weights_name, scaling))
