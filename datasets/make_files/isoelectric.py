'Storing isoelectric information in a dictionary'

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from enzynet.tools import dict_to_csv


# Initialization
isoelectric = {}

# Info from chapter 24 of Organic Chemistry (5th edition), by Wade
isoelectric['ALA'] = 6.0
isoelectric['ARG'] = 10.8
isoelectric['ASN'] = 5.4
isoelectric['ASP'] = 2.8
isoelectric['CYS'] = 5.0
isoelectric['GLU'] = 3.2
isoelectric['GLN'] = 5.7
isoelectric['GLY'] = 6.0
isoelectric['HIS'] = 7.6
isoelectric['ILE'] = 6.0
isoelectric['LEU'] = 6.0
isoelectric['LYS'] = 9.7
isoelectric['MET'] = 5.7
isoelectric['PHE'] = 5.5
isoelectric['PRO'] = 6.3
isoelectric['SER'] = 5.7
isoelectric['THR'] = 5.6
isoelectric['TRP'] = 5.9
isoelectric['TYR'] = 5.7
isoelectric['VAL'] = 6.0

# Center on neutral value
neutral_value = 6.0
isoelectric = {k: v - neutral_value for k, v in isoelectric.items()}

# Save results
dict_to_csv(isoelectric, '../isoelectric.csv')
