"""Amino-acid to weights mappings."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

# Info from p. 51 of The cell: a molecular approach, by Hausman et al.
CHARGE = {
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

# Values from J. Kyte et al. paper.
HYDROPATHY = {
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

# Info from chapter 24 of Organic Chemistry (5th edition), by Wade.
_RAW_ISOELECTRIC = {
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
_NEUTRAL_VALUE = 6.0
ISOELECTRIC = {amino_acid: raw_value - _NEUTRAL_VALUE
               for amino_acid, raw_value in _RAW_ISOELECTRIC.items()}
