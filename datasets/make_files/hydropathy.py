"""Storing hydropathy information in a dictionary."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from enzynet.tools import dict_to_csv

# Values from J. Kyte et al. paper.
hydropathy = {
    'ILE': 4.5,
    'VAL': 4.2,
    'LEU': 3.8,
    'PHE': 2.8,
    'CYS': 2.5,
    'MET': 1.9,
    'ALA': 1.8,
    'GLY': -0.4,
    'THR': -0.7,
    'TRP': -0.9,
    'SER': -0.8,
    'TYR': -1.3,
    'PRO': -1.6,
    'HIS': -3.2,
    'GLU': -3.5,
    'GLN': -3.5,
    'ASP': -3.5,
    'ASN': -3.5,
    'LYS': -3.9,
    'ARG': -4.5,
}

# Save results.
dict_to_csv(hydropathy, '../hydropathy.csv')
