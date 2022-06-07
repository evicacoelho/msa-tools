""" This program analyses a Multiple Sequence Alignment file, to give us informations about the conservation and correlation of the present sequences using Shannon's Information Theory. This program results in 5 graphic plots and a .xlsx file for analisys. """

# Libraries imported:

import os
import numpy as np
from Bio import AlignIO
from collections import OrderedDict as odict
from functions import *

# Constant defined:

# Working diretory
os.chdir('/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/')

if not os.path.exists('/home/earaujo/databases/HLA/results/'):
    os.makedirs('/home/earaujo/databases/HLA/results/')

# Alignments used:
N_A = 'HLA.last.fa'
N_B = 'CD8.last.fasta'
# Genome used:
START = 0
END = 5
ALPHABET = odict()
ALPHABET = {"A": 0, "R": 1, "N": 2, "D": 3, "Q": 4,
            "E": 5, "G": 6, "H": 7, "L": 8, "K": 9,
            "M":10, "F":11, "S":12, "T":13, "W":14,
            "Y":15, "C":16, "I":17, "P":18, "V":19,
            "-":20, ".":20, "B": 2, "Z": 4, "X":20,
            "J":20,}
Q = 21
# Error margin
LAMBDA = 0.001
THETA = 1.0
# Sequence size
OFFSET = 95

# Functions defined:

def point_show(result_a,result_b,a_cut=None,b_cut=None):
    higher = [];
    k_list = list(result_a.keys())
    if a_cut == None and b_cut == None:
        a_val = list(result_a.values()); b_val = list(result_b.values()) 
        a_cut = st.median(a_val); b_cut = st.median(b_val)
        for key in k_list:
            if result_a[key] >= a_cut:
                if result_b[key] >= b_cut:
                    higher.append(key) 
    else:
        a_cut = float(a_cut); b_cut = float(b_cut)
        if result_a[key] >= a_cut: 
            for key in k_list:
                if result_b[key] >= b_cut: 
                    higher.append(key)
    return higher


# Execution:

# Opens the alignment and transforms into a matrix.

A_handle = open(N_A, 'r'); B_handle = open(N_B, 'r');
msa_A = AlignIO.read(A_handle, 'fasta'); msa_B = AlignIO.read(B_handle, 'fasta')
seqnumber = len(msa_A)
seqlength = len(msa_A[0]) + len(msa_B[0])
encoded_msa0 = np.empty((seqnumber,seqlength),dtype=int)
seqs_b = encoded_msa0[:,OFFSET:]
for (x,i), A in np.ndenumerate(msa_A):
    encoded_msa0[x,i]=ALPHABET[A.upper()]
for (x,i), A in np.ndenumerate(msa_B):
    encoded_msa0[x,i+OFFSET]=ALPHABET[A.upper()]

all_mi = []
for index in range(START, END):    
    encoded_msa, Meff = Codemsa(np.load('genomes/genome.{}.npy'.format(index)), encoded_msa0, seqs_b, OFFSET, THETA)
    
    ics_a = list(range(OFFSET))
    ics_b = list(range(OFFSET,OFFSET + 108))
    ics = np.concatenate((ics_a, ics_b))

    nA = len(ics_a)
    nB = len(ics_b)
    seqlength = nA+nB

    pairs = []
    idx_pairs = []
    for i, ai in enumerate(ics_a):
        for j, aj in enumerate(ics_b):
            pairs.append((ai,aj))
            idx_pairs.append((i,j))

    nP = len(pairs)

    sitefreq = Sitefreq(encoded_msa, Meff, ics, nA+nB, Q, LAMBDA)
    pairfreq = Pairfreq(encoded_msa, Meff, ics, nA+nB, sitefreq, Q, LAMBDA)
    mi, h = information(sitefreq, pairfreq, nP, pairs, idx_pairs, Q)
    print('Genetration {}: Ʃ(MI) = {}; Ʃ(H) = {}.'.format(index,sum(mi), h))
    all_mi.append(sum(mi)); 
    
os.chdir('/home/earaujo/databases/HLA/sequences/')
np.save('trans_from_{}_to_{}'.format(START,END), all_mi)
