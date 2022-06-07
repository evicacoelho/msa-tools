""" Program to calculate the transinformation biased on the genetic algorithm acconting. """

# libraries

import os
import numpy as np
from Bio import SeqIO
from scipy import spatial
from sfunctions import *
from collections import OrderedDict as odict

# constants

START = 0
STOP = 1500
LAMBDA = 0.001
THETA = 1
OFFSET = 45
Q = 21
ALPHABET = {"A": 0, "R": 1, "N": 2, "D": 3, "Q": 4,
            "E": 5, "G": 6, "H": 7, "L": 8, "K": 9,
            "M":10, "F":11, "S":12, "T":13, "W":14,
            "Y":15, "C":16, "I":17, "P":18, "V":19,
            "-":20, ".":20, "B": 2, "Z": 4, "X":20, "J":20}

# files

SYSTEM = 'TAT-CXCR'
MSA_FIX = 'tat-edited-95.fasta'
MSA_VAR = 'cxc_genome_'
PATH_FIX = '/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa'
PATH_VAR = '/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/results/cxc_fasta'
RES_PATH = '{}/results/sgen'.format(PATH_FIX)

## execution

# encoding fixed

os.chdir(PATH_FIX)
encoded_fix, Meff_fix = codemsa(MSA_FIX, THETA, ALPHABET)
nFIX = len(encoded_fix[0])
# sitefreq
sitefreq_fix = sitefreq(encoded_fix, Meff_fix, nFIX, Q, LAMBDA)
# entropy
entropy_fix = entropy(sitefreq_fix, nFIX)

# encoding variable
os.chdir(PATH_VAR); 
gen_tMI = np.empty((STOP+1)); gen_MI = []
gen_Hvar = []

for i in range(START, STOP+1):
    msa_var = MSA_VAR+'{}.fasta'.format(i)
    encoded_var, Meff_var = codemsa(msa_var, THETA, ALPHABET)
    nVAR = len(encoded_var[0]); Meff = (Meff_fix+Meff_var)/2
    # sitefreq
    sitefreq_var = sitefreq(encoded_var, Meff_var, nVAR, Q, LAMBDA);
    #entropy
    entropy_var = entropy(sitefreq_var, nVAR)
    
    # conjoint analisys
    pairfr = pairfreq(encoded_fix, encoded_var, Meff, nFIX, nVAR, Q, LAMBDA)
    MI, tMI = mutualinfo(sitefreq_fix, sitefreq_var, pairfr, nFIX, nVAR)
    
    # returnings
    gen_tMI[i] = tMI; gen_MI.append(MI); gen_Hvar.append(entropy_var)
    print(' please wait... {}% concluded.'.format(round((i/STOP)*100, 3)))

# saves    
gen_MI = np.array((gen_MI), dtype=float); gen_Hvar = np.array((gen_Hvar), dtype=float)

if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)
os.chdir(RES_PATH)

np.save('sgenMI_{}.npy'.format(SYSTEM),gen_MI)
np.save('sgentMI_{}.npy'.format(SYSTEM),gen_tMI)
np.save('sgenHfix_{}.npy'.format(SYSTEM),entropy_fix)
np.save('sgenHvar_{}.npy'.format(SYSTEM),gen_Hvar)

print(' Done. Results returned in \n {}'.format(RES_PATH))
