#!/usr/bin/env python

"""UTILS.PY - Utility functions

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd                                                                                                                           

import os
import time
import numpy as np
import warnings
import gdown
from scipy import sparse
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln
import matplotlib.pyplot as plt
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
        
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# The doppler data directory
def datadir():
    """ Return the doppler data/ directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

# Split a filename into directory, base and fits extensions
def splitfilename(filename):
    """ Split filename into directory, base and extensions."""
    fdir = os.path.dirname(filename)
    base = os.path.basename(filename)
    exten = ['.fit','.fits','.fit.gz','.fits.gz','.fit.fz','.fits.fz']
    for e in exten:
        if base[-len(e):]==e:
            base = base[0:-len(e)]
            ext = e
            break
    return (fdir,base,ext)

def download_data(force=False):
    """ Download the data from my Google Drive."""

    # Check if the "done" file is there
    if os.path.exists(datadir()+'done') and force==False:
        return
    
    #https://drive.google.com/drive/folders/1SXId9S9sduor3xUz9Ukfp71E-BhGeGmn?usp=share_link
    # The entire folder: 1SXId9S9sduor3xUz9Ukfp71E-BhGeGmn
    
    data = [{'id':'17EDOUbzNr4cDzn7KZdPsw7R2lAj_o_np','output':'cannongrid_3000_18000_hotdwarfs_norm_cubic_model.pkl'},   
            {'id':'1w4CwoZsxEyBRs7DvrZ4P68PMBQuhAjav','output':'payne_coolhot_29.npz'}]
    
    # This should take 2-3 minutes on a good connection
    
    # Do the downloading
    t0 = time.time()
    print('Downloading '+str(len(data))+' Doppler data files')
    for i in range(len(data)):
        print(str(i+1)+' '+data[i]['output'])
        fileid = data[i]['id']
        url = f'https://drive.google.com/uc?id={fileid}'
        output = datadir()+data[i]['output']  # save to the data directory
        if os.path.exists(output)==False or force:
            gdown.download(url, output, quiet=False)

    print('All done in {:.1f} seconds'.format(time.time()-t0))
