"""Storing charge information in a dictionary."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from enzynet import tools

# Info from p. 51 of The cell: a molecular approach, by Hausman et al.
charge = {
    'ALA': 0,
    'ARG': 1,
    'ASN': 0,
    'ASP': -1,
    'CYS': 0,
    'GLU': -1,
    'GLN': 0,
    'GLY': 0,
    'HIS': 0.1,
    'ILE': 0,
    'LEU': 0,
    'LYS': 1,
    'MET': 0,
    'PHE': 0,
    'PRO': 0,
    'SER': 0,
    'THR': 0,
    'TRP': 0,
    'TYR': 0,
    'VAL': 0,
}

# Save results.
tools.dict_to_csv(charge, '../charge.csv')
