""" This program analyses a Multiple Sequence Alignment file, to give us informations about the conservation and correlation of the present sequences using Shannon's Information Theory. This program results in 5 graphic plots and a .xlsx file for analisys. """

# Libraries imported:

import os
import numpy as np
from Bio import SeqIO
from scipy import spatial
from sfunctions import *
from collections import OrderedDict as odict

# Constant defined:
SYSTEM = 'tat-cxcr'
LAMBDA = 0.001
THETA = 1
FILE_I = 'tat-edited-95.fasta'
FILE_J = "cxcr-edited.fasta"
PATH = "/home/earaujo/MEGA/data/host-pathogen-PPIN/msa"
RES_PATH = '/home/earaujo/MEGA/data/host-pathogen-PPIN/results/'.format(PATH)
AMINOS = odict()
AMINOS = {"A": 0, "R": 1, "N": 2, "D": 3, "Q": 4,
          "E": 5, "G": 6, "H": 7, "L": 8, "K": 9,
          "M":10, "F":11, "S":12, "T":13, "W":14,
          "Y":15, "C":16, "I":17, "P":18, "V":19,
          "-":20, ".":20, "B": 2, "Z": 4, "X":20, "J":20}

Q = 21

## Execution:

# Encoding

os.chdir(PATH)
encoded_i, Meff_i = codemsa(FILE_I, THETA, AMINOS); encoded_j, Meff_j= codemsa(FILE_J, THETA, AMINOS)
nI = len(encoded_i[0]); nJ = len(encoded_j[0])
Meff = (Meff_i+Meff_j)/2

print(' msa i: ({} lines, {} columns)'.format(len(encoded_i), nI))
print(' msa j: ({} lines, {} columns)'.format(len(encoded_j), nJ))
 
# Frequencies
sitefreq_i = sitefreq(encoded_i, Meff_i, nI, Q, LAMBDA)
sitefreq_j = sitefreq(encoded_j, Meff_j, nJ, Q, LAMBDA)

pairfreq = pairfreq(encoded_i, encoded_j, Meff, nI, nJ, Q, LAMBDA)

# Resultados

H_i = entropy(sitefreq_i, nI); H_j = entropy(sitefreq_j, nJ)
tH_i = np.sum(H_i); tH_j = np.sum(H_j)

MI, tMI = mutualinfo(sitefreq_i, sitefreq_j, pairfreq, nI, nJ)

# Retornos
os.chdir(RES_PATH)

np.save('H_'+FILE_I+'.npy', H_i); np.save('H_'+FILE_J+'.npy', H_j)
np.save('MI_'+SYSTEM+'.npy', MI)

print(' Done. Results returned in \n {}'.format(RES_PATH))
