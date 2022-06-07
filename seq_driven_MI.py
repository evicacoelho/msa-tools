""" This program uses one or a set of MSAs with different line positions to calculate transinformation. """

# libraries

import os
import numpy as np
import random as rd
import datetime as dtime
from Bio import AlignIO
from scipy import spatial
from joblib import Parallel, delayed
from functions import *

# constants

START = 0
STOP = 700
PATH_FIX = '/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/'
PATH_VAR = '/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/results'
PATH_RES = '/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/results'
MSA_FIX = "tat-edited-95.fasta"
LAMBDA = 0.001
OFFSET = 45
theta = 1.0 # Between 0.0 and 1.0
q = 21
aminos = {"A": 0, "R": 1, "N": 2, "D": 3, "Q": 4,
          "E": 5, "G": 6, "H": 7, "L": 8, "K": 9,
          "M":10, "F":11, "S":12, "T":13, "W":14,
          "Y":15, "C":16, "I":17, "P":18, "V":19,
          "-":20, ".":20, "B": 2, "Z": 4, "X":20, "J":20}

## execution

# ranges
ics_a = [2,14,15,16,20,24,25,27,28,32,33,34,47,52,76,79,81,82,90,94,99,100,101,104,106,110,111,112,113,116,119,120,122,123,124,125,128,129,130]
ics_b = list(range(OFFSET,OFFSET + 384))
ics = np.concatenate((ics_a, ics_b))

# looping
fitness = []
os.chdir(PATH_FIX)
fix_handle = open(MSA_FIX, "r")
for index in range(START,STOP+1):
    os.chdir(PATH_VAR)
    MSA_VAR = 'cxc_gen_{}.fasta'.format(index)
    # handles
    var_handle = open(MSA_VAR, "r")
    msa_fix = AlignIO.read(fix_handle, "fasta")
    msa_var = AlignIO.read(var_handle, "fasta")
    seqnumber = len(msa_fix)
    seqlength = len(msa_fix[0]) + len(msa_var[0])
    
    # encoding
    encoded_msa0=np.empty((seqnumber,seqlength),dtype=int)
    for (x,i), A in np.ndenumerate(msa_fix):
        encoded_msa0[x,i]=aminos[A.upper()]
    for (x,i), A in np.ndenumerate(msa_var):
        encoded_msa0[x,i+OFFSET]=aminos[A.upper()]

    seqs_b = encoded_msa0[:,OFFSET:]

    # defining sizes
    nA = len(ics_a)
    nB = len(ics_b)
    seqlength = nA+nB

    # defining contacts
    pairs = []
    idx_pairs = []
    for i, ai in enumerate(ics_a):
        for j, aj in enumerate(ics_b):
            pairs.append((ai,aj))
            idx_pairs.append((i,j))

    nP = len(pairs) # important for SP function

    encoded_msa, Meff = SCodemsa(MSA_VAR, aminos, OFFSET, theta)
    sitefreq = Sitefreq(encoded_msa, Meff, ics, nA+nB, q, LAMBDA)
    pairfreq = Pairfreq(encoded_msa, Meff, ics, nA+nB, sitefreq, q, LAMBDA)
    mi, h = information(sitefreq, pairfreq, nP, pairs, idx_pairs, q)
    
    fitness.append(np.sum(mi))

os.chdir(PATH_RES)

fil = open('{}_transinformation.dat'.format(MSA_FIX), 'w')
fil.write('{}'.format(fitness))
fil.close()
print(fitness)
