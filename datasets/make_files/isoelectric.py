"""Storing isoelectric information in a dictionary."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from enzynet.tools import dict_to_csv

# Info from chapter 24 of Organic Chemistry (5th edition), by Wade.
isoelectric = {
    'ALA': 6.0,
    'ARG': 10.8,
    'ASN': 5.4,
    'ASP': 2.8,
    'CYS': 5.0,
    'GLU': 3.2,
    'GLN': 5.7,
    'GLY': 6.0,
    'HIS': 7.6,
    'ILE': 6.0,
    'LEU': 6.0,
    'LYS': 9.7,
    'MET': 5.7,
    'PHE': 5.5,
    'PRO': 6.3,
    'SER': 5.7,
    'THR': 5.6,
    'TRP': 5.9,
    'TYR': 5.7,
    'VAL': 6.0,
}

# Center on neutral value.
neutral_value = 6.0
isoelectric = {k: v - neutral_value for k, v in isoelectric.items()}

# Save results.
dict_to_csv(isoelectric, '../isoelectric.csv')
