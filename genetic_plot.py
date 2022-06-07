''' Numpy general plotter. '''

# Libraries imported 

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects
from plot_ease import *
from collections import OrderedDict as odict

# User defined variables

gen = 3900;
output = 'image_result_TAT-CXCR.png'

# Changing working directory

os.chdir('/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/best-TAT_CXCR')

if not os.path.exists('/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/results'):
    os.makedirs('/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/results')
    
# Functions defined

# Program execution
gen_dict = odict()
for index in range(0, gen):
    name = 'best.{}.npy'.format(str(index))
    best = np.load(name)
    gen_dict[index] = best

keys = list(gen_dict.keys()); values = list(gen_dict.values());

fig, ax = splot(values, None, lw=1300,
                xname='Generations', yname='MI', tname='Transinformation per generation')

os.chdir('/home/earaujo/MEGA/databases/host-pathogen-PPIN/msa/results')
plt.savefig(output)
