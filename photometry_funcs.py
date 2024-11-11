# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:51:40 2024

@author: cbhof
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D
import glob as g
def centroid(x):
    '''
    

    Parameters
    ----------
    x : ARRAY
        photo array from a FITS file. Relies on numpy.

    Returns
    -------
    x_half = int x centroid value
    y_half = int y centroid value
    x_max = int upper range of x value
    x_min = int lower range of x value
    y_max = int upper range of x value
    y_min = int lower range of y value

    '''
    x_half = int(x.shape[1]/2)
    y_half = int(x.shape[0]/2)
    x_max = x_half + 30
    x_min = x_half - 30
    y_max = y_half + 30
    y_min = y_half - 30
    return(x_half, y_half, x_max, x_min, y_max, y_min)

def gauss2d(x, y, x_std, y_std, x_mean, y_mean):
    '''
    

    Parameters
    ----------
    x : array
        range of evenly spaced x values.
    y : array
        range of evenly spaced y values.
    x_std : int
        middle value of x pixel range.
    y_std : int
        middle value of y pixel range.
    x_mean : int
        centroid x value.
    y_mean : int
        centroid y value.

    Returns
    -------
    normalized :
        normalized 2 dimensional gaussian for contour plotting.

    '''
    x_num = (x - x_mean)**2
    y_num = (y-y_mean)**2
    xdenom = 2*(x_std**2)
    ydenom = 2*(y_std **2)
    gauss = np.exp(-((x_num/xdenom)+(y_num/ydenom)))
    normalized = gauss/np.sum(gauss)
    return(normalized)

def mag_instrument(n_net_star, t):
    '''
    

    Parameters
    ----------
    n_net_star : int
        N number of counts in a star image.
    t : int
        exposure time.

    Returns
    -------
    m_inst : int or float
        intstrument magnitude of star.

    '''
    m_inst = -2.5 * np.log10(n_net_star / t)
    return m_inst

def open_fits(filepath):
    '''
    depends on glob and astropy.io fits file handling.
    Opens fits files in filepath and stores data and headers.

    Parameters
    ----------
    filepath : str
        file path for the images being used.

    Returns
    -------
    images_arr : array
        image data from fits files. 
    images_header : array
        Header for each fits file.
    images : array
        shows files to tell which star corresponds to which image

    '''
    images = sorted(g.glob(filepath))
    images_arr = []
    images_header = []
    for i in range(len(images)):
        foo = fits.open(images[i])
        images_arr.append(foo[0].data)
        images_header.append(foo[0].header)
        foo.close()
    return images_arr, images_header, images