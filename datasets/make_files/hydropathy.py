"""Storing hydropathy information in a dictionary."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from enzynet.tools import dict_to_csv


# Initialization.
hydropathy = {}

# Values from J. Kyte et al. paper.
hydropathy['ILE'] = 4.5
hydropathy['VAL'] = 4.2
hydropathy['LEU'] = 3.8
hydropathy['PHE'] = 2.8
hydropathy['CYS'] = 2.5
hydropathy['MET'] = 1.9
hydropathy['ALA'] = 1.8
hydropathy['GLY'] = -0.4
hydropathy['THR'] = -0.7
hydropathy['TRP'] = -0.9
hydropathy['SER'] = -0.8
hydropathy['TYR'] = -1.3
hydropathy['PRO'] = -1.6
hydropathy['HIS'] = -3.2
hydropathy['GLU'] = -3.5
hydropathy['GLN'] = -3.5
hydropathy['ASP'] = -3.5
hydropathy['ASN'] = -3.5
hydropathy['LYS'] = -3.9
hydropathy['ARG'] = -4.5

# Save results.
dict_to_csv(hydropathy, '../hydropathy.csv')
