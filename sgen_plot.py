from plot_ease import *
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects
import os

# files

os.chdir('/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/results/sgen')

tMI = np.load('sgentMI_TAT-CXCR.npy'); 
MI = np.load('sgenMI_TAT-CXCR.npy')
Hv = np.load('sgenHvar_TAT-CXCR.npy');
Hf = np.load('sgenHfix_TAT-CXCR.npy')

MIi = np.empty((len(MI[0])), dtype=float)
for i in range(len(MI[0])):
    MIi[i] = np.sum(MI[:,i])
    
splot((MIi/Hf),Hf,note='b-')

os.chdir('/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/results/')

ref_MI = np.load('MI_TAT-CXCR.npy'); 
ref_MIi = np.empty((len(ref_MI)), dtype=float)
ref_Hf = np.load('H_tat-edited-95.fasta.npy')

for i in range(len(ref_MI)):
    ref_MIi[i] = np.sum(ref_MI[i,:])
    
def is_it_ics(a,b):
    if a*b >= 1130:
       print(' No')
    elif a*b < 1130:
       print(' yes')


