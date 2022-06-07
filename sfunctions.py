""" Set of simplified functions to give entropy and transinformation. """

# libraries

import numpy as np
from Bio import SeqIO
from scipy import spatial
from collections import OrderedDict as odict

# functions

def codemsa(fname, theta, ALPHABET):
    """ Transforms a given clustalx or fasta file onto a matrix based on a user defined alphabet. Also, returns the Meff. """
    psdmtx = []
    for record in SeqIO.parse(fname,'fasta'):
        sequence = record.seq
        line = list(sequence) 
        psdline = []
        for item in line:
            psdline.append(ALPHABET[item])
        psdmtx.append(psdline)
            
    encoded_msa = np.array(psdmtx).reshape(len(psdmtx),len(psdmtx[0]))
    
    hammdist = spatial.distance.pdist(encoded_msa, 'hamming')
    weight_matrix = spatial.distance.squareform(hammdist < (1.0- theta))
    weight = 1.0 / (np.sum(weight_matrix, axis = 1) + 1.0)
    Meff = np.sum(weight)
    return encoded_msa, Meff

def sitefreq(msa, Meff, nA, q, LAMBDA):
    """ Counts the frequence proportion in a column for a given matrix. """
    sitefreq = np.zeros((nA, q), dtype=float); #print(count)
    for i in range(len(msa)):
        for j in range(nA):
            for aa in range(q):
                if aa == msa[i,j]:
                    sitefreq[j,aa] += 1
                else:
                    sitefreq[j,aa] += 0
    
    sitefreq = sitefreq/Meff
    sitefreq = (1-LAMBDA)*sitefreq + LAMBDA/q
    return sitefreq
    

def pairfreq(msa_i, msa_j, Meff, nA, nB, q, LAMBDA):
    """ Counts the joint proportion in two columns for a given matrix. """
    pairfreq = np.zeros((nA,q,nB,q), dtype=float)
    for i in range(nA):
        for j in range(nB):
            for aa_i in range(q):
                for aa_j in range(q):
                    if msa_i[aa_i, i] == msa_j[aa_j, j]:
                        pairfreq[i,aa_i,j,aa_j] += 1
                    else:
                        pairfreq[i,aa_i,j,aa_j] += 0
    
    pairfreq /= Meff
    pairfreq = (1-LAMBDA)*pairfreq + LAMBDA/(q*q)
    return pairfreq
    
    
def entropy(freq, nA):
    """ Calculates the shannon entropy for a given value. """
    ent = np.zeros((nA),dtype=float)
    for i in range(0, nA):
        ent[i] = -np.sum(freq[i,:]*np.log(freq[i,:]))
    return ent
    
def mutualinfo(site_i, site_j, pairf, nA, nB):
    """ Calculates the transinformation between two proportions and the joint proportion. """
    MI = np.empty((nA, nB), dtype=float)
    for i in range(nA):
        for j in range(nB):
            MI[i,j] = np.sum(pairf[i,:,j,:]*np.log(pairf[i,:,j,:]/(site_i[i]*site_j[j])))
    
    tMI = np.sum(MI)
    return MI, tMI
