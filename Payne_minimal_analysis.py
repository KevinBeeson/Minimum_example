#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:37:08 2023

@author: kevin
"""
import os 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS']="1" 
os.environ['VECLIB_MAXIMUM_THREADS']="1"
import numpy as np
from scipy import signal
import logging
import scipy
from multiprocessing import Pool


def leaky_relu(z,grad=False):
    '''
    This is the activation function used by default in all our neural networks.
    '''
    limits=(z > 0)
    leaky=z*limits + 0.01*z*np.invert(limits)
    if grad:
        return leaky,limits
    return leaky

def payne_sythesize(solar_values,x_min,x_max,NN_coeffs,grad=False):
        """
        Synthesizes the spectra using Payne This takes alot of time
    
        Parameters
        ----------
        solar_values : a 1x8 array( the solar value arrays without vrad )  using teff,logg,monh,fe,alpha,vrad,vsini,vmac,vmic order
        x_min : min value for payne
        x_max : max values for payne
        NN_coeffs :Matrix coefficients gotten from Payne 
    
        Returns
        -------
        real_spec : 1xn array which is the payne sythesized spectra
    
        """
    
        scaled_labels = (solar_values-x_min)/(x_max-x_min) - 0.5
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
        if grad is False:

            inside =np.einsum('ij,j->i', w_array_0, scaled_labels)+ b_array_0
            inside1=np.einsum('ij,j->i', w_array_1, leaky_relu(inside))+ b_array_1
            # This takes the most time in the function
            real_spec=np.einsum('ij,j->i', w_array_2, leaky_relu(inside1))+ b_array_2
            return real_spec
def galah_kern(fwhm, b):
        """ Returns a normalized 1D kernel as is used for GALAH resolution profile """
        size=2*(fwhm/2.355)**2
        size_grid = int(size) # we limit the size of kernel, so it is as small as possible (or minimal size) for faster calculations
        if size_grid<7: size_grid=7
        x= scipy.mgrid[-size_grid:size_grid+1]
        g = np.exp(-0.693147*np.power(abs(2*x/fwhm), b))
        return g / np.sum(g)
def synth_resolution_degradation(wave_synth, synth,res_map,res_b,wave_original,l_new_premade=None,kernel_=None,synth_res=300000.0,grad=None):
        """
        this takes alot of time 
        Take a synthetic spectrum with a very high  resolution and degrade its resolution to the resolution profile of the observed spectrum. The synthetic spectrum should not be undersampled, or the result of the convolution might be wrong.
        Parameters:
            synth np array or similar: an array representing the synthetic spectrum. Must have size m x 2. First column is the wavelength array, second column is the flux array. Resolution of the synthetic spectrum must be constant and higher than that of the observed spectrum.
            synth_res (float): resolving power of the synthetic spectrum
        Returns:
            Convolved syntehtic spectrum as a np array of size m x 2.
        """
        synth=np.vstack((wave_synth,synth)).T
        
        l_original=wave_synth
        #check if the shape of the synthetic spectrum is correct
        if synth.shape[1]!=2: logging.error('Syntehtic spectrum must have shape m x 2.')

        #check if the resolving power is high enough
        sigma_synth=synth[:,0]/synth_res
        # if max(sigma_synth)>=min(res_map)*0.95: logging.error('Resolving power of the synthetic spectrum must be higher.')

        #check if wavelength calibration of the synthetic spectrum is linear:
        if abs((synth[:,0][1]-synth[:,0][0])-(synth[:,0][-1]-synth[:,0][-2]))/abs(synth[:,0][1]-synth[:,0][0])>1e-6:
            logging.error('Synthetic spectrum must have linear (equidistant) sampling.')        

        #current sampling:
        sampl=galah_sampl=synth[:,0][1]-synth[:,0][0]
        galah_sampl=wave_original[1]-wave_original[0]


        #original sigma
        s_original=sigma_synth





        #oversampling. If synthetic spectrum sampling is much finer than the size of the kernel, the code would work, but would return badly sampled spectrum. this is because from here on the needed sampling is measured in units of sigma.
        oversample=galah_sampl/sampl*5.0

        #minimal needed sampling

        #keep adding samples until end of the wavelength range is reached
        if l_new_premade is None:
            #required sigma (resample the resolution map into the wavelength range of the synthetic spectrum)
            s_out=np.interp(synth[:,0], wave_original, res_map)


            #the sigma of the kernel is:
            s=np.sqrt(s_out**2-s_original**2)

            #fit it with the polynomial, so we have a function instead of sampled values:
            map_fit=np.poly1d(np.polyfit(synth[:,0], s, deg=6))

            #create an array with new sampling. The first point is the same as in the spectrum:
            l_new=[synth[:,0][0]]

            min_sampl=max(s_original)/sampl/sampl*oversample

            l_new=np.array(numba_syth_resolution(map_fit.coef,l_new,sampl,min_sampl,synth[:,0][-1]))
            kernel_=galah_kern(max(s_original)/sampl*oversample, res_b)

        else:
            l_new=l_new_premade
        # while l_new[-1]<synth[:,0][-1]+sampl:
        #     l_new.append(l_new[-1]+map_fit(l_new[-1])/sampl/min_sampl)

        #interpolate the spectrum to the new sampling:
        new_f=np.interp(l_new,synth[:,0],synth[:,1])

        # This takes the most time in the function
        con_f=signal.fftconvolve(new_f,kernel_,mode='same')

        #inverse the warping:
        synth[:,1]=np.interp(l_original,l_new,con_f)
        if l_new_premade is None:
            return synth[:,1],l_new,kernel_
        if not grad is None:
            new_grad=[np.interp(np.array(l_new),synth[:,0],x) for x in grad]
            con_grad=[signal.fftconvolve(x,kernel_,mode='same') for x in new_grad]
            grad=[np.interp(l_original,np.array(l_new),x) for x in con_grad]
            return synth[:,1],grad

        return synth[:,1]
# @jit(nopython=True,parallel=False,cache=True)
def numba_syth_resolution(coefficients,l_new,sampl,min_sampl,last_frequency):
        """
        Slow part of resolution degradation is done in JIT to make it faster
    
        Parameters
        ----------
        coefficients : TYPE
            DESCRIPTION.
        l_new : TYPE
            DESCRIPTION.
        sampl : TYPE
            DESCRIPTION.
        min_sampl : TYPE
            DESCRIPTION.
        last_frequency : TYPE
            DESCRIPTION.
    
        Returns
        -------
        l_new : TYPE
            DESCRIPTION.
    
        """
        while l_new[-1]<last_frequency+sampl:
            poly=sum([l_new[-1]**power*x for power,x in enumerate(coefficients[::-1])])
                                     
            l_new.append(l_new[-1]+poly/sampl/min_sampl)
        
        return l_new

#Creates some fake data and models
wavelength=np.linspace(4000, 4500,4096)
res_map=np.ones(4096)*0.23+np.random.normal(0,0.1,4096)
res_b=2.263


#Creates a fake neural network model
w_array_0 = np.random.normal(0,0.01,size=(300,36))
w_array_1 = np.random.normal(0,0.01,size=(300,300))
w_array_2 =  np.random.normal(0,0.01,size=(50750,300))
b_array_0 = np.random.normal(0,0.01,size=(300))
b_array_1 = np.random.normal(0,0.01,size=(300))
b_array_2 =np.random.normal(0,0.01,size=(50750))
x_min = np.ones(36)*-1
x_max = np.ones(36)
NN_coeffs= (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
wavelength_synthetic=np.linspace(3900, 4600,50750)
l_new,kernel_=synth_resolution_degradation(wavelength_synthetic,np.ones(50750),res_map,res_b,wavelength)[1:]


#creates some fake data to input to the neural network model 

number_of_iteration=1000 #change me for the number iterations
all_input_data=np.random.normal(0.0,0.1,size=(number_of_iteration,36))

import time

#The function where all the slow down happens
def example_loop(input_data):
    #In this function the slow part is the np.einsum
    synthetic_spectra=payne_sythesize(input_data, x_min, x_max, NN_coeffs)
    #in this function the slow part is the signal.fftconvolve
    synthetic_spectra=synth_resolution_degradation(wavelength_synthetic,synthetic_spectra,res_map,res_b,wavelength,l_new,kernel_)


ncpu=1
t0=time.time()
#just added pool as was playing with changing the numbers of cpus I dont get much of an imporvement above 6 and sometimes the cpu makes it worse
with Pool(processes=ncpu) as pool:
    pool.map(example_loop,all_input_data)

t1=time.time()
print("took " +str(t1-t0)+ " s for "+str(number_of_iteration) +" iterations")