'Get characteristics from PDB files'

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import os.path
import urllib.request
import warnings

import numpy as np

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import *

from enzynet.tools import read_dict, scale_dict


warnings.filterwarnings("ignore", category=PDBConstructionWarning)

backbone_ids = ['C', 'N', 'CA']

current_directory = os.path.dirname(os.path.abspath(__file__))
PDB_path = os.path.join(current_directory, '../files/PDB/')
datasets_path = os.path.join(current_directory, '../datasets/')

class PDB_backbone(object):
    """
    Functions aimed at information extraction from PDB files.


    Parameters
    ----------
    pdb_id : string
        ID of the desired PDB file, e.g. '102L'.

    path : string (optional, default is 'files/PDB')
        Path where the PDB file is located, or where the PDB file should be
        downloaded.

    """
    def __init__(self, pdb_id, path=PDB_path):
        'Initialization'
        self.pdb_id = pdb_id.upper()
        fullfilename = os.path.join(path, pdb_id.lower() + '.pdb')
        if os.path.isfile(fullfilename): # File in directory
            pass
        else:
            urllib.request.urlretrieve('http://files.rcsb.org/download/' +
                                        pdb_id.upper() + '.pdb',
                                        fullfilename)
        self.structure = PDBParser().get_structure(pdb_id.upper(), fullfilename)

    def get_coords(self):
        'Gets coordinates of backbone'
        # Initialization
        backbone_coords = []
        backbone_atoms = []

        # Computations
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=True): # Check if amino acid
                        for atom in residue:
                            if atom.get_name() in backbone_ids:
                                backbone_coords = backbone_coords + [atom.get_coord().tolist()]
                                backbone_atoms = backbone_atoms + [atom.get_name()]

        # Store results
        self.backbone_coords = np.array(backbone_coords)
        self.backbone_atoms = backbone_atoms

    def get_coords_extended(self, p=5):
        'Adds the coordinates from interpolation betweens atoms of the backbone'
        # Initialization
        self.get_coords()
        C_coords = self.__get_coords_specific_atom(backbone_ids)
        new_coords = np.zeros((p*(C_coords.shape[0]-1), C_coords.shape[1]))

        # Computations
        for i in range(1, C_coords.shape[0]):
            for k in range(p, 0, -1):
                new_coords[p*i-k,:] = ((p-k+1)*C_coords[i-1,:] + k*C_coords[i,:])/(p+1)

        # Store results
        self.backbone_coords_ext = np.concatenate((self.backbone_coords, new_coords), axis=0)

    def get_weights(self, weights='hydropathy', scaling=True):
        'Gets weights of each position regarding associated amino-acid'
        # Initialization
        backbone_weights = []
        backbone_residues = []

        # Select weights
        weights = read_dict(os.path.join(datasets_path, weights + '.csv'))
        weights = dict([key, float(value)] for key, value in weights.items()) # Convert values to float

        # Scaling
        if scaling == True:
            weights = scale_dict(weights)

        # Computations
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=True): # Check if standard amino acid
                        local_residue = residue.get_resname()
                        local_weight = weights[local_residue]
                        for atom in residue:
                            if atom.get_name() in backbone_ids:
                                backbone_weights = backbone_weights + [local_weight]
                                backbone_residues = backbone_residues + [local_residue]

        # Store results
        self.backbone_weights = backbone_weights
        self.backbone_residues = backbone_residues

    def get_weights_extended(self, p, weights='hydropathy', scaling=True):
        'Adds the weights from interpolation between atoms of the backbone'
        # Initialization
        self.get_weights(weights=weights, scaling=scaling)
        C_weights = self.__get_weights_specific_atom(backbone_ids)
        new_weights = np.zeros((p*(len(C_weights)-1)))

        # Computations
        for i in range(1, len(C_weights)):
            for k in range(p, 0, -1):
                new_weights[p*i-k] = ((p-k+1)*C_weights[i-1] + k*C_weights[i])/(p+1)

        # Store results
        self.backbone_weights_ext = np.concatenate((self.backbone_weights, new_weights), axis=0)

    def __get_coords_specific_atom(self, specific_atoms):
        'Extracts coordinates of a specific atom only'
        for atom in specific_atoms: # Tries to accomplish the task on the first that works
            try:
                return np.array([self.backbone_coords[i,:]
                                 for i in range(self.backbone_coords.shape[0])
                                 if self.backbone_atoms[i] in specific_atoms])
            except:
                pass

    def __get_weights_specific_atom(self, specific_atoms):
        'Extracts weights of a specific atom only'
        for atom in specific_atoms: # Tries to accomplish the task on the first that works
            try:
                return [self.backbone_weights[i]
                        for i in range(len(self.backbone_weights))
                        if self.backbone_atoms[i] in specific_atoms]
            except:
                pass

    def get_ligands(self):
        'Stores ligands in a list'
        # Initialization
        ligands = []

        # Find ligands
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    ligands += [residue.get_resname()]

        # Store unique IDs
        self.ligands = list(set(ligands))
