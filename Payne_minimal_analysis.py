
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 13:39:38 2021

@author: kevin
"""
import os 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS']="1" 
os.environ['VECLIB_MAXIMUM_THREADS']="1"
from scipy.stats import kde

from numba import jit
from functools import  partial
from astropy.io.votable import parse
import emcee
import scipy
from scipy import signal
from os.path import exists
import subprocess
from pathlib import Path
import os.path
import logging
from astropy.table import Table,vstack
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from astropy.io import fits
import copy
import functools
from multiprocessing.dummy import Pool as ThreadPool 
import warnings

#ignore by message
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
warnings.filterwarnings("ignore")

def shift_maker(solar_value,given_labels=['teff','logg','fe_h','fe_h','alpha','vrad_Blue','vrad_Green','vrad_Red','vrad_IR','vsini','vmac','vmic'],fe_hold=False,all_labels=['teff','logg','fe_h','fe_h','alpha','vrad_Blue','vrad_Green','vrad_Red','vrad_IR','vsini','vmac','vmic']):
        """
        Create a dictionary of solar parameters from an array of values
        
    
        Parameters
        ----------
        solar_values : array of 9 length in the order of  teff,logg,monh,fe,alpha,vrad,vsini,vmac,vmic
    
        Returns
        -------
        shift : dictionary of solar parameters
    
        """
        shift={}
        skip=0
        for value,x in enumerate(all_labels):
            if x in given_labels:
                if x=='fe_h' and fe_hold:
                    shift[x]=shift['fe_h']
                else:
                    shift[x]=solar_value[value-skip]
            else:
                skip+=1
                bands=spectras.bands
                colour=[y for y in bands if y in x]
                if x=='fe_h' and fe_hold:
                    shift[x]=shift['fe_h']
                elif not 'vrad' in x:

                    shift[x]=rgetattr(spectras,bands[0]+'.'+x)

                elif len(colour): 
                    shift[x]=rgetattr(spectras, colour[0]+'.vrad')
        return shift


def log_prior(shift):
        """
        Gives the log prior of the current star given the photometric teff and logg 
    
        Parameters
        ----------
        shift : dictionary of the solar parameters 
    
        Returns
        -------
        float of logg of the prior.
    
        """
        error=0
        for x in shift:
            if x in ('logg','teff'):
                error-=(shift[x]-float(old_abundances[x]))**2/(2*(float(old_abundances['e_'+x]))**2)
        if error<-1e100:
            return -np.inf
        else:
            return error        
#@
def log_posterior(solar_values,parameters,prior=False,full_parameters=None,insert_mask=None,create_limit_mask=True,create_masks=True,):
    
        """
        Gives how good the fit of the inserted solar parameters are compared to the observed spectra 
    
        Parameters
        ----------
        solar_values : 1x9 array of solar parameters with teff,logg,monh,fe,alpha,vrad,vsini,vmac,vmic  order
            DESCRIPTION.
        prior : BOOL, optional
            if True it takes into account the photometric prior. The default is True.
        parameters: list of parameters that youre using
        full paramters: all the parameters in the model
        first try : is the mask
    
        Returns
        -------
        TYPE
        float of logg of how good the fit is .
    
        """
        if full_parameters==None:
            full_parameters=parameters
        #Sanity check to see if the solar values are in a certain section
        if len(solar_values)!= len(parameters):
            print('youre missing parameters')
        
        if not starting_test(solar_values,spectras.old_abundances,parameters,cluster=True):
              return -np.inf
        # print('passed')
        
        shift=shift_maker(solar_values,given_labels=parameters,all_labels=full_parameters)

        # synthesezes new spectra the main things that take time in this function is synth_resolution_degradation (fftconvole) and Payne_synthesize
        synthetic_spectras=spectras.synthesize(shift,give_back=True)
        normalized_spectra=[rgetattr(spectras,x+'.spec') for x in spectras.bands]
        normalized_uncs=[rgetattr(spectras,x+'.uncs') for x in spectras.bands]
    

        if insert_mask is None:
            if create_limit_mask==True:
                normalized_limit_array=spectras.limit_array(give_back=True,observed_spectra=normalized_spectra)
            else:
                normalized_limit_array=[np.ones(len(x)) for x in normalized_spectra]
            if create_masks:
                normalized_masks=spectras.create_masks(clean=True,shift=shift)
            else:
                normalized_masks=[np.ones(len(x)) for x in normalized_spectra]
            combined_mask=[x*y for x,y in zip(normalized_masks,normalized_limit_array)]
        else:
            combined_mask=insert_mask
        probability=spectras.log_fit(synthetic_spectra=synthetic_spectras,solar_shift=shift,normal=normalized_spectra,uncertainty=normalized_uncs,limit_array=combined_mask,combine_masks=False)
        # print(prior_2d(shift))
        if prior:
            probability+=prior_2d(shift)
        if probability>0:
            print('oh no prob ',probability)
            print(repr(solar_values))
            # return False
        # print('probability', probability)
        return probability
@jit(nopython=True,cache=True)
def normal(x,mu,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2)
def prior_2d(shift):
    """
    Creates a prior from a 2d probabilty distribution gotten from photometric temperatures and logg

    Parameters
    ----------
    shift :DICT
        a dictionary that has temperature and logg which we would like to calculate the probability of it using the prior.

    Returns
    -------
    FLOAT
        log of the probability.

    """
    #Spread number defines if the sampling of the prior has been fine enough. if it has we use a sumation method(spread number is bellow 1) 
    #and if it hasnt we use a kde method (spread number above 1)
    if spectras.spread_number>1:
        #calculates the kde if hasnt been done
        if spectras.kde==None:
            spectras.kde=kde.gaussian_kde([old_abundances['teff_raw'],old_abundances['logg_raw']])
        photometric_probability_density=spectras.kde
        e_teff=spectras.e_teff_photometric
        e_logg=spectras.e_logg_photometric
        # f= lambda teff_var,logg_var,teff_0,logg_0:photometric_probability_density([teff_var,logg_var])*normal(teff_0,teff_var,e_teff)*normal(logg_0,logg_var,e_logg)
        total_int=0
        teff_prior=shift['teff']
        #applies the shift from spectroscopic temperature to photometric
        polynomial_coeff=old_abundances['coeff']
        spectroscopic_shift=np.poly1d(polynomial_coeff)
        teff_prior+=spectroscopic_shift(teff_prior)
        
        logg_prior=shift['logg']
        # teff_line=np.linspace(teff_prior-e_teff*10, teff_prior+e_teff*10,2)
        # logg_line=np.linspace(logg_prior-e_logg*10, logg_prior+e_logg*10,2)
        
        total_int=photometric_probability_density([teff_prior,logg_prior])[0]
        # for x in range(len(teff_line)-1):
        #     for y in range(len(logg_line)-1):
        #         total_int+=integrate.dblquad(f, logg_line[y], logg_line[y+1], lambda teff_var: teff_line[x], lambda teff_var:teff_line[x+1],args=(teff_prior,logg_prior))[0]
    else:
        # teff_raw=fast_abundances['teff_raw']
        # logg_raw=old_abundances['logg_raw']
        # e_teff_raw=old_abundances['e_teff_raw']
        # e_logg_raw=old_abundances['e_logg_raw']
        
        teff_raw=fast_abundances[0]
        logg_raw=fast_abundances[1]
        e_teff_raw=fast_abundances[2]
        e_logg_raw=fast_abundances[3]

        prior_parameters=np.column_stack((teff_raw,e_teff_raw,logg_raw,e_logg_raw))
        teff_prior=shift['teff']
        logg_prior=shift['logg']
        #applied the shift from spectroscopic temperature to photometric

        # polynomial_coeff=fast_abundances[5]
        spectroscopic_shift=np.poly1d(fast_coefficients)

        teff_prior+=spectroscopic_shift(teff_prior)

        total_int=0
        for temp_param in prior_parameters:           
            total_int+=normal(teff_prior,temp_param[0],temp_param[1])*normal(logg_prior,temp_param[2],temp_param[3])
        total_int/=len(old_abundances['teff_raw'])
    #if the probability is zero it means that the asked parameters are too far away from the photometric temperatures. 
    #Thus we have to use a simple normal approximation
    if total_int==0:
        e_teff=spectras.e_teff_photometric
        e_logg=spectras.e_logg_photometric
        return np.log(1/(e_teff*e_logg*2*np.pi))-(1/2*((teff_prior-old_abundances['teff_photometric'])/e_teff)**2)-\
            (1/2*((logg_prior-old_abundances['logg_photometric'])/e_logg)**2)
    
    return np.log(total_int)
            
            
#Three function so one can set and get attributed in classes in mutiple levels i.e. spectras.Blue.wave
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
def runGetattr(obj, attr):
    def _getattr(obj, attr):
        return getattr(obj, attr)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
def starting_test(solar_values,old_abundances,parameters=['teff','logg','fe_h','fe_h','alpha','vrad_Blue','vrad_Green','vrad_Red','vrad_IR','vsini','vmac','vmic'],cluster=False):
        """
        Sanity check to see if the solar values inserted passes a sanity check
    
        Parameters
        ----------
        solar_values : 1x9 array with the order of teff,logg,monh,fe,alpha,vrad,vsini,vmac,vmic
    
        Returns
        -------
        bool
            If it doesnt pass the sanity check it will return False and if passes will return True
    
        """
        shift={}
        for y,x in zip(parameters,solar_values):
            shift[y]=x
        labels_with_limits=['teff','logg','fe_h','vmic','vsini','Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
        labels_with_limits=[x for x in labels_with_limits if x in parameters]
        # elem_big_dips=['Li','K']
        # elem_big_dips=[x for x in elem_big_dips if x in parameters]
        for x in labels_with_limits:
            if shift[x]<x_min[x]-abs(x_min[x])*0.2 or shift[x]>x_max[x]*1.2:
                print('outside of the models limits ',x,' value ', shift[x])
                return False
        # for x in elem_big_dips:
        #     mean=old_abundances[x.lower()+'_fe']
        #     sig=old_abundances['e_'+x.lower()+'_fe']
        #     if shift[x]<mean-sig*3 or shift[x]>mean+sig*3:
        #         print('outside of the dip limits',x)
        #         return False
        vrad=['vrad_Blue','vrad_Green','vrad_Red','vrad_IR']
        vrad=[x for x in vrad if x in parameters]
        vrad_r=['rv'+x[4:]+'_r' for x in vrad]
        radial_velocities=[shift[x] for x in vrad]
        if len (radial_velocities):
            for rad,rad_r in zip(vrad,vrad_r):
                if not np.isnan(old_abundances['red_rv_ccd'][vrad.index(rad)]):
                    mean=float(old_abundances['red_rv_ccd'][vrad.index(rad)])
                elif not np.isnan(old_abundances['red_rv_com']):
                    mean=float(old_abundances['red_rv_com'])
                else:
                    break
                if not np.isnan(old_abundances['red_e_rv_ccd'][vrad.index(rad)]):
                    sig=float(old_abundances['red_e_rv_ccd'][vrad.index(rad)])*3
    
                elif not np.isnan(old_abundances['red_e_rv_com']):
                    sig=float(old_abundances['red_e_rv_com'])*3
                else:
                    sig=5
                if abs(shift[rad]-mean)>sig+2:
                    print(rad,' is wrong by ',str(float(abs(shift[rad]-old_abundances['red_rv_com']))))
                    return False
            if np.count_nonzero(~np.isnan(radial_velocities))>1.0:
                if max(radial_velocities)-min(radial_velocities)>(np.nanmax(radial_velocities)-np.nanmin(radial_velocities))*4:
                    print('radial velocities too different')
                    return False
            else:
                if max(radial_velocities)-min(radial_velocities)>5.0:
                    print('radial velocities too different')
                    return False

        return True


def galah_kern(fwhm, b):
        """ Returns a normalized 1D kernel as is used for GALAH resolution profile """
        size=2*(fwhm/2.355)**2
        size_grid = int(size) # we limit the size of kernel, so it is as small as possible (or minimal size) for faster calculations
        if size_grid<7: size_grid=7
        x= scipy.mgrid[-size_grid:size_grid+1]
        g = scipy.exp(-0.693147*np.power(abs(2*x/fwhm), b))
        return g / np.sum(g)
def dopler(original_wavelength,v_rad,synthetic_wavelength,synthetic_spectra,grad=None):
        """
        Shifts and crops your spectra to be the same length as the observed spectra
    
        Parameters
        ----------
        original_wavelength : 1xn array of the observed wavelength range (i.e. observed wavelengths)
        spectra : 1xm synthesized spectra that you want to shift
        v_rad : float
            the radial velocity you want to shift your spectra by.
        Returns
        -------
        observed_wavelength : TYPE
            shifted wavelength
    
        """
        c=299792.458
        delta_v=v_rad/c
        #my spectra are syntheszied to be larger than the galah spectra so now it crops see by how much 
        # if len(original_wavelength)!=len(spectra) and synthetic_wavelength is None:
        #     difference=abs(len(original_wavelength)-len(spectra))
        #     original_wavelength_strenched=np.interp(np.linspace(-difference/2,difference/2+len(original_wavelength),num=len(spectra)),range(len(original_wavelength)),original_wavelength)
            
        #     observed_wavelength=original_wavelength_strenched*(1+delta_v)
        if not synthetic_wavelength is None:
            observed_wavelength=synthetic_wavelength*(1+delta_v)
        else:
            observed_wavelength=original_wavelength*(1+delta_v)
        observed_wavelength_linear=np.linspace(observed_wavelength[0],observed_wavelength[-1],num=len(observed_wavelength))
        synthetic_spectra_shifted_linear=np.interp(observed_wavelength_linear,observed_wavelength,synthetic_spectra)
        return observed_wavelength,synthetic_spectra_shifted_linear
        # #crops and interpolates 
        # spectra_new=np.interp(original_wavelength,observed_wavelength,spectra)
        # if not grad is None:
        #     grad_new=[np.interp(original_wavelength,observed_wavelength,x) for x in grad]
        #     return spectra_new,grad_new
        # return spectra_new

@jit(nopython=True,parallel=False,cache=True)
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



class individual_spectrum:
    
    def __init__(self,name,interpolation,x,count,old_abundances,cluster=True,starting_values='dr4'):
            limits={'Blue':[4705,4908],'Green':[5643,5879],'Red':[6470,6743],'IR':[7577.0,7894.0]}
            # spacings={'Blue':0.004,'Green':0.005,'Red':0.006,'IR':0.008}
            if x=='IR':
                starting_fraction=1000/4096
                length_fraction=3096/(4096*(1-starting_fraction))
            elif x=='Red':
                starting_fraction=0/4096
                length_fraction=4096/(4096*(1-starting_fraction))
            elif x=='Green':
                starting_fraction=0/4096
                length_fraction=4096/(4096*(1-starting_fraction))
            elif x=='Blue':
                starting_fraction=0/4096
                length_fraction=4096/(4096*(1-starting_fraction))
            else:
                starting_fraction=2085/4096
                length_fraction=2/(4096*(1-starting_fraction))
            tmp = np.load("NN_normalized_spectra_all_elements_"+x+".npz")
            w_array_0 = tmp["w_array_0"]
            w_array_1 = tmp["w_array_1"]
            w_array_2 = tmp["w_array_2"]
            b_array_0 = tmp["b_array_0"]
            b_array_1 = tmp["b_array_1"]
            b_array_2 = tmp["b_array_2"]
            x_min = tmp["x_min"]
            x_max = tmp["x_max"]
            tmp.close()
            NN_coeffs= (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
            self.grad=(w_array_0, w_array_1, w_array_2)
            self.NN_coeff=NN_coeffs
            self.x_min=x_min
            self.x_max=x_max
            Path('dr6.1/'+name[0:6]+'/spectra/com/').mkdir(parents=True,exist_ok=True)
            try:
                hermes=fits.open('dr6.1/'+name[0:6]+'/spectra/com/'+name+str(count)+'.fits')
                x0= float(hermes[1].header.get('CRVAL1'))
                x1=float( hermes[1].header.get('CRVAL1')+len(hermes[1].data)* hermes[1].header.get('CDELT1'))
                fstart= x0+(x1-x0)*starting_fraction
                
                new_start=int(starting_fraction*len(hermes[1].data))
                new_end=new_start+int(len(hermes[1].data)*(1-starting_fraction)*length_fraction)
                
                
                length=np.linspace(x0+(x1-x0)*starting_fraction,x1,num=int(4096*(1-starting_fraction)))
                length_synthetic=np.linspace(limits[x][0], limits[x][1],num=len(b_array_2))
                fend=length[-1]
                #Increases the length of synthetic spectra so it over interpolates  
                self.wave_synth=length_synthetic
                self.wave=length
                self.spec=hermes[0].data[new_start:new_end]
                self.spec_original=hermes[0].data[new_start:new_end]

                self.uncs=hermes[2].data[new_start:new_end]
                self.uncs*=self.spec[0]
                self.uncs_original=copy.deepcopy(self.uncs)
                hermes[0].data=hermes[0].data[new_start:new_end]
                hermes[1].data=hermes[1].data[new_start:new_end]
                hermes[2].data=hermes[2].data[new_start:new_end]
                hermes[7].data=hermes[7].data[new_start:new_end]
                self.wran=[fstart,fend]
                self.hermes=hermes
                if starting_values=='dr4':
                    if cluster:
                        if not (np.isnan(old_abundances['teff_spectroscopic']) or np.ma.is_masked(old_abundances['teff_spectroscopic'])):
                            self.teff=float(old_abundances['teff_spectroscopic'])
                        else:
                            self.teff=float(old_abundances['teff_photometric'])
                        if not( np.isnan(old_abundances['vmic']) or np.ma.is_masked(old_abundances['vmic'])):

                            self.vmic=float(old_abundances['vmic'])
                        elif not (np.isnan(old_abundances['red_vmic']) or np.ma.is_masked(old_abundances['red_vmic'])):
                            self.vmic=float(old_abundances['red_vmic'])
                        else:
                            self.vmic=1
                        if not( np.isnan(old_abundances['vsini']) or np.ma.is_masked(old_abundances['vsini'])):

                            self.vsini=float(old_abundances['vsini'])
                        elif not (np.isnan(old_abundances['red_vbroad']) or np.ma.is_masked(old_abundances['red_vbroad'])):
                            self.vsini=float(old_abundances['red_vbroad'])
                        else:
                            self.vsini=10
                            
                        if not( np.isnan(old_abundances['fe_h']) or np.ma.is_masked(old_abundances['fe_h'])):

                            self.fe_h=float(old_abundances['fe_h'])
                        elif not (np.isnan(old_abundances['red_fe_h']) or  np.ma.is_masked(old_abundances['red_fe_h'])):
                            self.fe_h=float(old_abundances['red_fe_h'])
                        else:
                            self.fe_h=0.0
                            
                        if not (np.isnan(old_abundances['red_rv_ccd'][count-1]) or np.ma.is_masked(old_abundances['red_rv_ccd'][count-1])) :
                            self.vrad=old_abundances['red_rv_ccd'][count-1]
                        elif not (np.isnan(float(old_abundances['red_rv_com']))or np.ma.is_masked(old_abundances['red_rv_com'])):
                            self.vrad=float(old_abundances['red_rv_com'])
                        else:
                            self.vrad=0.0
                        self.logg=float(old_abundances['logg_photometric'])
                        elements=['Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
                        for individual_element in elements:
                            if not isinstance(old_abundances[individual_element.lower()+'_fe'],np.float32):
                                setattr(self,individual_element,0.0)
                            else:
                                setattr(self,individual_element,old_abundances[individual_element.lower()+'_fe'])
    
                        
                    else:
                        self.teff=float(old_abundances['teff_spectroscopic'])
                        self.vmic=float(old_abundances['vmic'])
                        self.vsini=float(old_abundances['vsini'])
                        self.Fe=float(old_abundances['fe_h'])
                        self.fe_h=float(old_abundances['fe_h'])
                        if not np.isnan(old_abundances['red_rv_ccd'][count-1]):
                            self.vrad=old_abundances['red_rv_ccd'][count-1]
                        elif not np.isnan(float(old_abundances['red_rv_com'])):
                            self.vrad=float(old_abundances['red_rv_com'])
                        else:
                            self.vrad=0.0
                        self.vmac=6.0
                        if not np.isnan(old_abundances['logg_spectrometric']):
                            self.logg=float(old_abundances['logg_spectrometric'])
                        else:
                            self.logg=float(old_abundances['logg_photometric'])

                        self.monh=float(old_abundances['fe_h'])
                        elements=['Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
                        for individual_element in elements:
                            if not isinstance(old_abundances[individual_element.lower()+'_fe'],np.float32):
                                setattr(self,individual_element,0.0)
                            else:
                                setattr(self,individual_element,old_abundances[individual_element.lower()+'_fe'])
                else:
                    self.teff=float(old_abundances['teff_'+starting_values])
                    self.vmic=float(old_abundances['vmic_'+starting_values])
                    self.vsini=float(old_abundances['vsini_'+starting_values])
                    self.Fe=float(old_abundances['fe_h_'+starting_values])
                    self.fe_h=float(old_abundances['fe_h_'+starting_values])
                    if not np.isnan(old_abundances['red_rv_ccd'][count-1]):
                        self.vrad=old_abundances['red_rv_ccd'][count-1]
                    elif not np.isnan(float(old_abundances['red_rv_com'])):
                        self.vrad=float(old_abundances['red_rv_com'])
                    else:
                        self.vrad=0.0
                    self.vmac=6.0
                    self.logg=float(old_abundances['logg_'+starting_values])
                    self.monh=float(old_abundances['fe_h_'+starting_values])
                    elements=['Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
                    for individual_element in elements:
                        if not isinstance(old_abundances[individual_element+'_Fe_'+starting_values],np.float):
                            setattr(self,individual_element,0.0)
                        else:
                            setattr(self,individual_element,old_abundances[individual_element+'_Fe_'+starting_values])

                if hermes[1].header.get('TEFF_R')=='None' or hermes[1].header.get('LOGG_R')=='None':
                    print('Warning the reduction didnt produce an TEFF spectra might not reduce properly')
                    print('enter to continue')
                    self.bad_reduction=True
                else:
                    self.bad_reduction=False
                line, wavelength = np.loadtxt('important_lines',usecols=(0,1),unpack=True,dtype=str, comments=';')
                np.append(wavelength,'4861.3230')
                np.append(wavelength,'6562.7970')
                np.append(line,r'H$_\beta$')
                np.append(line,r'H$_\alpha$')
                wavelength=wavelength.astype(float)
                important_lines_temp=np.vstack([[elem_temp,wave_temp] for elem_temp,wave_temp in zip(line,wavelength) if wave_temp>length_synthetic[0] and wave_temp<length_synthetic[-1]])
                self.important_lines=important_lines_temp
                rsetattr(self,'.important_lines',important_lines_temp)
                self.normal_value=None
    
                self.l_new=None
                # Load spectrum masks
                masks = Table.read('spectrum_mask_kevin.fits')
                masks_temp=vstack([x for x in masks if x['mask_begin']>self.wran[0]-200 and x['mask_end']<self.wran[1]+200])
                self.masks=masks_temp
                vital_lines = Table.read('vital_lines.fits')
                vital_lines_temp=vstack([x for x in vital_lines if x['line_begin']>self.wran[0]-200 and x['line_end']<self.wran[1]+200])
                self.vital_lines=vital_lines_temp

            except FileNotFoundError:
                print('No hermes found')
                self.hermes=None
                self.bad_reduction=True
            



def sclip(p,fit,n,ye=[],sl=99999,su=99999,min=0,max=0,min_data=1,grow=0,global_mask=None,verbose=True):
    """
    p: array of coordinate vectors. Last line in the array must be values that are fitted. The rest are coordinates.
    fit: name of the fitting function. It must have arguments x,y,ye,and mask and return an array of values of the fitted function at coordinates x
    n: number of iterations
    ye: array of errors for each point
    sl: lower limit in sigma units
    su: upper limit in sigma units
    min: number or fraction of rejected points below the fitted curve
    max: number or fraction of rejected points above the fitted curve
    min_data: minimal number of points that can still be used to make a constrained fit
    global_mask: if initial mask is given it will be used throughout the whole fitting process, but the final fit will be evaluated also in the masked points
    grow: number of points to reject around the rejected point.
    verbose: print the results or not
    """

    nv,dim=np.shape(p)

    #if error vector is not given, assume errors are equal to 0:
    if ye==[]: ye=np.zeros(dim)
    #if a single number is given for y errors, assume it means the same error is for all points:
    if isinstance(ye, (int, float)): ye=np.ones(dim)*ye

    if global_mask==None: global_mask=np.ones(dim, dtype=bool)
    else: pass

    f_initial=fit(p,ye,global_mask)
    s_initial=np.std(p[-1]-f_initial)

    f=f_initial
    s=s_initial

    tmp_results=[]

    b_old=np.ones(dim, dtype=bool)

    for step in range(n):
        #check that only sigmas or only min/max are given:
        if (sl!=99999 or su!=99999) and (min!=0 or max!=0):
            raise RuntimeError('Sigmas and min/max are given. Only one can be used.')

        #if sigmas are given:
        if sl!=99999 or su!=99999:
            b=np.zeros(dim, dtype=bool)
            if sl>=99999 and su!=sl: sl=su#check if only one is given. In this case set the other to the same value
            if su>=99999 and sl!=su: su=sl

            good_values=np.where(((f-p[-1])<(sl*(s+ye))) & ((f-p[-1])>-(su*(s+ye))))#find points that pass the sigma test
            b[good_values]=True

        #if min/max are given
        if min!=0 or max!=0:
            b=np.ones(dim, dtype=bool)
            if min<1: min=dim*min#detect if min is in number of points or percentage
            if max<1: max=dim*max#detect if max is in number of points or percentage

            bad_values=np.concatenate(((p[-1]-f).argsort()[-int(max):], (p[-1]-f).argsort()[:int(min)]))
            b[bad_values]=False

        #check the grow parameter:
        if grow>=1 and nv==2:
            b_grown=np.ones(dim, dtype=bool)
            for ind,val in enumerate(b):
                if val==False:
                    ind_l=ind-int(grow)
                    ind_u=ind+int(grow)+1
                    if ind_l<0: ind_l=0
                    b_grown[ind_l:ind_u]=False

            b=b_grown

        tmp_results.append(f)

        #check that the minimal number of good points is not too low:
        if len(b[b])<min_data:
            step=step-1
            b=b_old
            break

        #check if the new b is the same as old one and break if yes:
        if np.array_equal(b,b_old):
            step=step-1
            break

        #fit again
        f=fit(p,ye,b&global_mask)
        s=np.std(p[-1][b]-f[b])
        b_old=b

    if verbose:
        print('')
        print('FITTING RESULTS:')
        print('Number of iterations requested:    ',n)
        print('Number of iterations performed:    ', step+1)
        print('Initial standard deviation:        ', s_initial)
        print('Final standard deviation:          ', s)
        print('Number of rejected points:         ',len(np.invert(b[np.invert(b)])))
        print('')

    return f,tmp_results,b
@jit(nopython=True,parallel=True,cache=True)
def array_limiter(mother_array,baby_array_1,baby_array_2,limit=1.05):
    """
    returns the both arrays where the value of the mother_array is bellow 1.05

    Parameters
    ----------
    mother_array : 1xn array
    baby_array : 1xn array
        DESCRIPTION.
    limit : float, optional
        whats the cut of limit. The default is 1.05.

    Returns
    -------
    2 arrays 
        both arrays where the mother array's value is bellow 1.05.

    """
    all_arrays=np.column_stack((mother_array,baby_array_1,baby_array_2))
    all_temp=[]
    [all_temp.append(x) for x in all_arrays if x[0]<1.05]
    all_temp=np.array(all_temp)
    return all_temp[:,0],all_temp[:,1],all_temp[:,2]
class spectrum_all:
    # bands=['IR']

    def __init__(self,input_data,interpolation=10,cluster=True,bands=None,starting_values='dr4'):
        if isinstance(input_data,str) or isinstance(input_data,int) or np.issubdtype(input_data,np.integer):
            name=str(input_data)
        elif isinstance(input_data,Table.Row):
            name=str(input_data['sobject_id'])
            old_abundances=input_data
        else:
            print('needs to be either a string sobject id string or a astropy row of the star')
        if bands==None:
            bands=['Blue']
        self.rv_shift=1e-10
        self.bands=bands
        self.name=name
        self.interpolation=interpolation
        self.sister_stars=None
        self.starting_values=starting_values
        if cluster:
            mask=photometric_data['sobject_id']==np.int64(name)
            photometric_prior_information=photometric_data[mask]
            photometric_prior_information=photometric_prior_information[0]
            spread_temperature=np.sqrt(np.var(photometric_prior_information['teff_raw']))
            sig_teff=np.mean(photometric_prior_information['e_teff_raw'])
            spread_logg=np.sqrt(np.var(photometric_prior_information['logg_raw']))
            sig_logg=np.mean(photometric_prior_information['e_logg_raw'])
            self.spread_number=spread_temperature*spread_logg/(sig_logg*sig_teff*len(photometric_prior_information['e_logg_raw']))
            self.e_teff_photometric=sig_teff
            self.e_logg_photometric=sig_logg
            self.kde=None
        if isinstance(input_data,Table.Row):
            starting_data=old_abundances
        elif cluster:
            starting_data=photometric_prior_information
        self.old_abundances=starting_data

        for count,x in enumerate(bands,1):
            setattr(self,x,individual_spectrum(name,interpolation,x,count,starting_data,cluster,starting_values=starting_values))
        bands_new=[]
        for x in bands:
            if rgetattr(self, x+'.hermes') is not None:
                bands_new.append(x)
        bands=bands_new
        self.bands=bands_new
        self.correct_resolution_map()
        self.equilize_spectras()
    def equilize_spectras(self,colours=None):
        if colours==None:
            colours=self.bands
        for x in colours:
            if rgetattr(self, x+'.hermes')!=None:
                wavelength=rgetattr(self, x+'.wave')
                hermes=rgetattr(self, x+'.hermes')
                resmap=hermes[7].data
                observed_spectra=rgetattr(self, x+'.spec')
                observed_error=rgetattr(self, x+'.uncs')
                # new_observed,new_error=equalize_resolution(wavelength,resmap,observed_spectra,observed_error,hermes)
                rsetattr(self, x+'.spec',observed_spectra)
                rsetattr(self, x+'.uncs',observed_error)
                rsetattr(self, x+'.spec_equalized',observed_spectra)
                rsetattr(self, x+'.uncs_equalized',observed_error)
                
    def limit_array(self,colours=None,limit=1.05,observed_spectra=None,give_back=False):
        if colours==None:
            colours=self.bands
        if give_back:
            returning_limit=np.array(np.ones(len(colours)),dtype=object)
            if not np.array_equal(observed_spectra,None):
                for value,spec in enumerate(observed_spectra):
                    limit_array_temp=[]
                    for y in spec:
                        if y>limit:
                            limit_array_temp.append(0)
                        else:
                            limit_array_temp.append(1)
                    limit_array_spread=spread_masks(limit_array_temp,5)
                    returning_limit[value]=limit_array_spread
                return returning_limit
            for value,x in enumerate(colours):
                
                if rgetattr(self, x+'.hermes')!=None:
                    limit_array=[]
                    for y in rgetattr(self,x+'.spec'):
                        if y>limit:
                            limit_array.append(0)
                        else:
                            limit_array.append(1)
                    limit_array_spread=spread_masks(limit_array,5)
                    returning_limit[value]=limit_array_spread
            return returning_limit
        for x in colours:
            if not np.array_equal(observed_spectra,None):
                for value,spec in enumerate(observed_spectra):
                    limit_array_temp=[]
                    for y in spec:
                        if y>limit:
                            limit_array_temp.append(0)
                        else:
                            limit_array_temp.append(1)
                    limit_array_spread=spread_masks(limit_array_temp,5)
                    rsetattr(self,x+'.limit',limit_array_spread)

            if rgetattr(self, x+'.hermes')!=None:
                limit_array=[]
                for y in rgetattr(self,x+'.spec'):
                    if y>limit:
                        limit_array.append(0)
                    else:
                        limit_array.append(1)
                limit_array_spread=spread_masks(limit_array,5)
                rsetattr(self,x+'.limit',limit_array_spread)
            
    def solar_value_maker(self,shift,colours=None,keys=['teff','logg','fe_h','vmic','vsini','Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']):
        if colours==None:
            colours=self.bands

        solar_values=[]
        for x in keys:
            if x in shift:
                solar_values.append(shift[x])
            else:
                solar_values.append(rgetattr(self,colours+'.'+x))
        return solar_values
    def hermes_checker(self,limit=20,colours=None):
        if colours==None:
            colours=self.bands
        for x in colours:
            spectra=rgetattr(self, x+'.spec')
            if min(spectra)<0:
                print (min(spectra),' ',x)
                return True
        return False
    def create_masks(self,colours=None,synthetic_spectra_insert=None,uncs_insert=None,normalized_observed_spectra_insert=None,shift=None,clean=False,limits=[5,0.3]):
        if colours==None:
            colours=self.bands

        masks_all=np.array(np.ones(len(colours)),dtype=object)
        #Clean is a mask without the difference between synthetik spectra and observed
        if clean:
            masks_all=np.array(np.ones(len(colours)),dtype=object)

            for value,x in enumerate(colours):
                
                masks=rgetattr(self,x+'.masks')
                vital_lines=rgetattr(self,x+'.vital_lines')
                original_wavelength_strenched=rgetattr(self,x+'.wave')
                if not shift is None and 'vrad_'+x in shift:
                    v_rad=shift['vrad_'+x]

                else:
                    v_rad=rgetattr(self,x+'.vrad')
                c=299792.458
                delta_v=v_rad/c
                wave_opt=original_wavelength_strenched*(1-delta_v)
                masks_temp2=[]
                masks_temp=(
                    (
                        (~np.any(np.array([((wave_opt >= mask_beginning) & (wave_opt <= mask_end)) for (mask_beginning, mask_end) in zip(masks['mask_begin'],masks['mask_end'])]),axis=0))
                    ) |
                    # or is in vital line wavelengths
                    np.any(np.array([((wave_opt >= line_beginning) & (wave_opt <= line_end)) for (line_beginning, line_end) in zip(vital_lines['line_begin'],vital_lines['line_end'])]),axis=0)
                )
                for y in masks_temp:
                    if y:
                        masks_temp2.append(1)
                    else:
                        masks_temp2.append(0)
                masks_all[value]=masks_temp2
            return masks_all

        if  np.array_equal(synthetic_spectra_insert,None):
            for value,x in enumerate(colours):
                synthetic_spectra=rgetattr(self,x+'.synth')
                observed_spectra=rgetattr(self,x+'.spec')
                uncs=rgetattr(self,x+'.uncs')
                masks=rgetattr(self,x+'.masks')
                vital_lines=rgetattr(self,x+'.vital_lines')
                original_wavelength_strenched=rgetattr(self,x+'.wave')
                v_rad=rgetattr(self,x+'.vrad')
                c=299792.458
                delta_v=v_rad/c
                wave_opt=original_wavelength_strenched*(1-delta_v)
                masks_temp2=[]
                masks_bad_spectra=(
                    (
                        # Not too large difference between obs and synthesis
                        (~((np.abs(synthetic_spectra-observed_spectra)/uncs > limits[0]) & (np.abs(synthetic_spectra-observed_spectra) > limits[1])))
                        # Not in unreliable synthesis region
                    ) |
                    # or is in vital line wavelengths
                    np.any(np.array([((wave_opt >= line_beginning) & (wave_opt <= line_end)) for (line_beginning, line_end) in zip(vital_lines['line_begin'],vital_lines['line_end'])]),axis=0)
                )
                
                masks_bad_spectra_temp=[]
                for y in masks_bad_spectra:
                    if y:
                        masks_bad_spectra_temp.append(1)
                    else:
                        masks_bad_spectra_temp.append(0)
                masks_bad_spectra_spread=spread_masks(masks_bad_spectra_temp,3)
                # masks_bad_spectra_spread=np.ones(len(masks_bad_spectra_spread))-masks_bad_spectra_spread
                masks_bad_spectra_spread=[bool(x) for x in masks_bad_spectra_spread]
                overall_masks=(
                    (
                        # Not too large difference between obs and synthesis
                        (np.array(masks_bad_spectra_spread))&
                        # Not in unreliable synthesis region
                        (~np.any(np.array([((wave_opt >= mask_beginning) & (wave_opt <= mask_end)) for (mask_beginning, mask_end) in zip(masks['mask_begin'],masks['mask_end'])]),axis=0))
                    ) |
                    # or is in vital line wavelengths
                    np.any(np.array([((wave_opt >= line_beginning) & (wave_opt <= line_end)) for (line_beginning, line_end) in zip(vital_lines['line_begin'],vital_lines['line_end'])]),axis=0)
                )
                overall_masks_temp=[]
                for y in overall_masks:
                    if y:
                        overall_masks_temp.append(1)
                    else:
                        overall_masks_temp.append(0)
                rsetattr(self, x+'.masked_area',overall_masks_temp)
                limits_temp=[]
                limits_first_loop=[]
                first=True
                for wave_temp,mask_temp in zip(wave_opt,overall_masks_temp):
                    if not mask_temp:
                        if first:
                            limits_first_loop.append(wave_temp)
                            first=False
                    else:
                        if not first:
                            limits_first_loop.append(wave_temp)
                            limits_temp.append(limits_first_loop)
                            limits_first_loop=[]
                            first=True
                rsetattr(self, x+'.masked_limits',limits_temp)
                    
        else:
            masks_all=np.array(np.ones(len(colours)),dtype=object)

            for value,x in enumerate(colours):
        
                synthetic_spectra=synthetic_spectra_insert[value]
                observed_spectra=normalized_observed_spectra_insert[value]
                uncs=uncs_insert[value]
                masks=rgetattr(self,x+'.masks')
                vital_lines=rgetattr(self,x+'.vital_lines')
                original_wavelength_strenched=rgetattr(self,x+'.wave')
                v_rad=shift['vrad_'+x]
                c=299792.458
                delta_v=v_rad/c
                wave_opt=original_wavelength_strenched*(1-delta_v)
                masks_temp2=[]
                masks_bad_spectra=(
                    (
                        # Not too large difference between obs and synthesis
                        (~((np.abs(synthetic_spectra-observed_spectra)/uncs > limits[0]) & (np.abs(synthetic_spectra-observed_spectra) > limits[1])))
                        # Not in unreliable synthesis region
                    ) |
                    # or is in vital line wavelengths
                    np.any(np.array([((wave_opt >= line_beginning) & (wave_opt <= line_end)) for (line_beginning, line_end) in zip(vital_lines['line_begin'],vital_lines['line_end'])]),axis=0)
                )
                
                masks_bad_spectra_temp=[]
                for y in masks_bad_spectra:
                    if y:
                        masks_bad_spectra_temp.append(1)
                    else:
                        masks_bad_spectra_temp.append(0)
                masks_bad_spectra_spread=spread_masks(masks_bad_spectra_temp,3)
                # masks_bad_spectra_spread=np.ones(len(masks_bad_spectra_spread))-masks_bad_spectra_spread
                masks_bad_spectra_spread=[bool(x) for x in masks_bad_spectra_spread]
                overall_masks=(
                    (
                        # Not too large difference between obs and synthesis
                        (np.array(masks_bad_spectra_spread))&
                        # Not in unreliable synthesis region
                        (~np.any(np.array([((wave_opt >= mask_beginning) & (wave_opt <= mask_end)) for (mask_beginning, mask_end) in zip(masks['mask_begin'],masks['mask_end'])]),axis=0))
                    ) |
                    # or is in vital line wavelengths
                    np.any(np.array([((wave_opt >= line_beginning) & (wave_opt <= line_end)) for (line_beginning, line_end) in zip(vital_lines['line_begin'],vital_lines['line_end'])]),axis=0)
                )
                overall_masks_temp=[]
                for y in overall_masks:
                    if y:
                        overall_masks_temp.append(1)
                    else:
                        overall_masks_temp.append(0)
                masks_all[value]=overall_masks_temp
            return masks_all
    #@profile
    def synthesize(self,shift={},colours=None,multi=False,give_back=False,full=False,grad=False):
        if colours==None:
            colours=self.bands

        if full and not give_back:
            print('you probably want the spectrum back run again with give_back=True')
            return False
        if not give_back:
            if not multi:
                for x in colours:       
                    solar_values=self.solar_value_maker(shift,x)
                    dopler_shifted_spectra = payne_sythesize(solar_values,rgetattr(self,x+'.x_min'),rgetattr(self,x+'.x_max'),rgetattr(self,x+'.NN_coeff'))
                    if 'vrad_'+x in shift:
                        dopler_shifted_synth_wave,spectrum_shifted=dopler(rgetattr(self,x+'.wave'),shift['vrad_'+x],synthetic_wavelength=rgetattr(self,x+'.wave_synth'),synthetic_spectra=dopler_shifted_spectra)
                    else:
                        dopler_shifted_synth_wave,spectrum_shifted=dopler(rgetattr(self,x+'.wave'),rgetattr(self,x+'.vrad'),synthetic_wavelength=rgetattr(self,x+'.wave_synth'),synthetic_spectra=dopler_shifted_spectra)
                    if rgetattr(self,x+'.l_new') is None:
                        dopler_shifted_spectra,l_new,kernel=synth_resolution_degradation(
                            wave_synth=dopler_shifted_synth_wave,
                            synth=spectrum_shifted,
                            res_map=rgetattr(self,x+'.hermes')[7].data,
                            res_b=rgetattr(self,x+'.hermes')[7].header['b'],
                            wave_original=rgetattr(self,x+'.wave')
                            )
                        rsetattr(self,x+'.l_new',l_new)
                        rsetattr(self,x+'.kernel',kernel)
                    else:
                        dopler_shifted_spectra=synth_resolution_degradation(
                            wave_synth=dopler_shifted_synth_wave,
                            synth=spectrum_shifted,
                            res_map=rgetattr(self,x+'.hermes')[7].data,
                            res_b=rgetattr(self,x+'.hermes')[7].header['b'],
                            wave_original=rgetattr(self,x+'.wave'),
                            l_new_premade=rgetattr(self,x+'.l_new'),
                            kernel_=rgetattr(self,x+'.kernel'))
                    dopler_shifted_spectra=scipy.interpolate.CubicSpline(dopler_shifted_synth_wave,dopler_shifted_spectra)(rgetattr(self,x+'.wave'))
                    rsetattr(self,x+'.synth',dopler_shifted_spectra)
            else:
                with Pool(4) as pool:
                    inputs=[(shift,x,False,False) for x in colours]
                    pool.map(getattr(self,'synthesize'),inputs)
                # with ThreadPool(4) as pool:
                #     inputs=[(shift,x,False,False) for x in colours]
                #     pool.map(getattr(self,'synthesize'),inputs)
                
        else:
            if not multi:
                if grad:
                    
                    returning_spectra=np.array(np.ones(len(colours)),dtype=object)
                    returning_grad=np.array(np.ones(len(colours)),dtype=object)
                    for number,x in enumerate(colours):       
                        solar_values=self.solar_value_maker(shift,x)

                        spectrum,grad_spec = payne_sythesize(solar_values,rgetattr(self,x+'.x_min'),rgetattr(self,x+'.x_max'),rgetattr(self,x+'.NN_coeff'),grad=True)
                        if full:
                            returning_spectra[number]=spectrum
                            continue   
                        if 'vrad_'+x in shift:
                            dopler_shifted_spectra,grad_spec=dopler(rgetattr(self,x+'.wave'),spectrum,shift['vrad_'+x],synthetic_wavelength=rgetattr(self,x+'.wave_synth'),grad=grad_spec)
                        else:
                            dopler_shifted_spectra,grad_spec=dopler(rgetattr(self,x+'.wave'),spectrum,rgetattr(self,x+'.vrad'),synthetic_wavelength=rgetattr(self,x+'.wave_synth'),grad=grad_spec)

                        # if rgetattr(self,x+'.l_new') is None:
                            
                        #     dopler_shifted_spectra,l_new,kernel,grad_dopler=synth_resolution_degradation(rgetattr(self,x+'.wave'),dopler_shifted_spectra,rgetattr(self,x+'.hermes')[7].data,rgetattr(self,x+'.hermes')[7].header['b'],rgetattr(self,x+'.wave'),grad=grad_spec)
                        #     rsetattr(self,x+'.l_new',l_new)
                        #     rsetattr(self,x+'.kernel',kernel)

                        # else:
                        #     dopler_shifted_spectra,grad_dopler=synth_resolution_degradation(rgetattr(self,x+'.wave'),dopler_shifted_spectra,rgetattr(self,x+'.hermes')[7].data,rgetattr(self,x+'.hermes')[7].header['b'],rgetattr(self,x+'.wave'),rgetattr(self,x+'.l_new'),rgetattr(self,x+'.kernel'),grad=grad_spec)
                        # returning_grad[number]=grad_dopler
                        returning_spectra[number]=dopler_shifted_spectra
                    return returning_spectra,returning_grad
                else:
                    returning_spectra=[[0] for x in range(len(colours))]
                    for number,x in enumerate(colours):       
                        solar_values=self.solar_value_maker(shift,x)
    
                        dopler_shifted_spectra = payne_sythesize(solar_values,rgetattr(self,x+'.x_min'),rgetattr(self,x+'.x_max'),rgetattr(self,x+'.NN_coeff'))
                        if full:
                            returning_spectra[number]=spectrum
                            continue                   
                        if 'vrad_'+x in shift:
                            dopler_shifted_synth_wave,spectrum_shifted=dopler(rgetattr(self,x+'.wave'),shift['vrad_'+x],synthetic_wavelength=rgetattr(self,x+'.wave_synth'),synthetic_spectra=dopler_shifted_spectra)
                        else:
                            dopler_shifted_synth_wave,spectrum_shifted=dopler(rgetattr(self,x+'.wave'),rgetattr(self,x+'.vrad'),synthetic_wavelength=rgetattr(self,x+'.wave_synth'),synthetic_spectra=dopler_shifted_spectra)
                        if rgetattr(self,x+'.l_new') is None:
                            dopler_shifted_spectra,l_new,kernel=synth_resolution_degradation(
                                wave_synth=dopler_shifted_synth_wave,
                                synth=spectrum_shifted,
                                res_map=rgetattr(self,x+'.hermes')[7].data,
                                res_b=rgetattr(self,x+'.hermes')[7].header['b'],
                                wave_original=rgetattr(self,x+'.wave')
                                )
                            rsetattr(self,x+'.l_new',l_new)
                            rsetattr(self,x+'.kernel',kernel)
                        else:
                            dopler_shifted_spectra=synth_resolution_degradation(
                                wave_synth=dopler_shifted_synth_wave,
                                synth=spectrum_shifted,
                                res_map=rgetattr(self,x+'.hermes')[7].data,
                                res_b=rgetattr(self,x+'.hermes')[7].header['b'],
                                wave_original=rgetattr(self,x+'.wave'),
                                l_new_premade=rgetattr(self,x+'.l_new'),
                                kernel_=rgetattr(self,x+'.kernel'))
                        dopler_shifted_spectra=scipy.interpolate.CubicSpline(dopler_shifted_synth_wave,dopler_shifted_spectra)(rgetattr(self,x+'.wave'))
                        returning_spectra[number]=dopler_shifted_spectra
                    return returning_spectra
            else :
                with ThreadPool() as pool:
                    inputs=[(shift,x,False,True) for x in colours]
                    return pool.map(partial(self.synthesize_multi,shift=shift),colours)
                    # pool.map(partial(getattr(self,'synthesize_multi'),give_back=True,shift=shift),colours=colours)
    def synthesize_multi(self,colours,shift):
        return self.synthesize(shift,multi=False,give_back=True,colours=[colours])
    def correct_resolution_map(self,colours=None):
        if colours==None:
            colours=self.bands
        name=self.name
        not_resolved=[np.int64(name)]
        correct_spectra=[]
        # print(not_resolved)

        for count,x in enumerate(colours,1):
            if rgetattr(self,x+'.hermes')!=None:
                hermes=rgetattr(self,x+'.hermes')
                if len(hermes[7].data)!=len(hermes[1].data) or min(hermes[7].data)<0.02:
                    correct_spectra.append(0)
                    tried=True
                    print('Resolution map of ',name +str(count),' is wrong')
                    pivot_sisters=[]
                    temp_data=[x for x in all_reduced_data if x['sobject_id']==np.int64(name)]
                    temp_data=temp_data[0]
                    plate=temp_data['plate']
                    epoch=temp_data['epoch']
                    pivot_number=temp_data['pivot']
                    min_difference=0
                    while len(hermes[7].data)!=len(hermes[1].data) or min(hermes[7].data)<=0.02:
                        if self.sister_stars==None or tried==False:
    
                            if pivot_number!=0 and min_difference*365<15 and len(not_resolved)<15:
                                if len(pivot_sisters)==0:
                                    pivot_sisters=[x for x in all_reduced_data if x['pivot']==pivot_number and x['plate']==plate and not x['sobject_id'] in not_resolved and abs(x['epoch']-epoch)<0.1 and x['res'][count-1]>0]
                                else:
                                    pivot_sisters=[x for x in pivot_sisters if x['pivot']==pivot_number and x['plate']==plate and not x['sobject_id'] in not_resolved and abs(x['epoch']-epoch)<0.1 and x['res'][count-1]>0]
                                if len(pivot_sisters)>0:
                                    pivot_sisters=vstack(pivot_sisters)
                                    # difference=difference_UT_vector(time, pivot_sisters['utdate'])
                                    pivot_sisters['difference']=abs(pivot_sisters['epoch']-epoch)
                                    pivot_sisters.sort('difference')
                                    name_target=str(pivot_sisters[0]['sobject_id'])
                                    min_difference=pivot_sisters[0]['difference']
                                    # print('here')
                                    print('copying from '+name_target+' for '+ name)
                                    not_resolved.append(np.int64(name_target))
                                    # print(not_resolved)
                                    Path('dr6.1/'+name_target[0:6]+'/spectra/com/').mkdir(parents=True,exist_ok=True)
                                    if not exists('dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits'):
                                        #source='/media/storage/HERMES_REDUCED/dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits'
                                        source='kevin@gigli.fmf.uni-lj.si:/media/storage/HERMES_REDUCED/dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits'
                    
                                        destination='dr6.1/'+name_target[0:6]+'/spectra/com/'
                                        subprocess.run(["rsync",'-av',source,destination])
                                    try:
                                        if os.path.getsize('dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits')>380000:
                                            hermes_temp=fits.open('dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits')
                                            if x=='IR':
                                                try:
                                                    hermes[7].data=hermes_temp[7].data[-2048:]
                                                except TypeError:
                                                    print('cant be openned')
                                            else:
                                                try:
                                                    hermes[7].data=hermes_temp[7].data
                                                except TypeError:
                                                    print('cant be openned')
                                        else:
                                            print('file is too small')
                                    except FileNotFoundError:
                                        print('hermes_error')
                                else:
                                    print('no more sister stars ', name)
                                    min_difference=30
                            else:
                                print('pivot is ',pivot_number,' setting res_map to 0.4 A and the difference is ',min_difference*365)
                                hermes[7].data=0.40*(np.ones(len(hermes[1].data)))
                                name_target=None
    
    
    
                        elif tried:
                            tried=False
                            name_target=self.sister_stars
                            try:
                                Path('dr6.1/'+name_target[0:6]+'/spectra/com/').mkdir(parents=True,exist_ok=True)
                                if not exists('dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits'):
                                    #source='/media/storage/HERMES_REDUCED/dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits'
                                    source='kevin@gigli.fmf.uni-lj.si:/media/storage/HERMES_REDUCED/dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits'
                
                                    destination='dr6.1/'+name_target[0:6]+'/spectra/com/'
                                    subprocess.run(["rsync",'-av',source,destination])
        
                                if os.path.getsize('dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits')>380000:
                                    hermes_temp=fits.open('dr6.1/'+name_target[0:6]+'/spectra/com/'+name_target+str(count)+'.fits')
                                    if x=='IR':
                                        try:
                                            hermes[7].data=hermes_temp[7].data[-2048:]
                                        except TypeError:
                                            print('cant be openned')
                                    else:
                                        try:
                                            hermes[7].data=hermes_temp[7].data
                                        except TypeError:
                                            print('cant be openned')
                                else:
                                    print('file is too small')
                            except FileNotFoundError:
                                print('file not found')
                    self.sister_stars=name_target
                else:
                    correct_spectra.append(1)
            else:
                correct_spectra.append(0)
        self.resolution_spectra=correct_spectra
    def normalize(self,colours=None,data=None):
        if colours==None:
            colours=self.bands
        len_colours=range(len(colours))
        returning_spectra=[[0] for x in len_colours]
        returning_uncs=[[0] for x in len_colours]
        for value,x in enumerate(colours):
            if not np.array_equal(data,None):
                
                
                
                original_line=rgetattr(self,x+'.spec_equalized')
                x_line=rgetattr(self,x+'.wave')
                synth_line=data[value]
                uncertainty=rgetattr(self,x+'.uncs_equalized')
                
                renormalisation_fit = sclip((x_line,original_line/synth_line),chebyshev,int(3),ye=uncertainty,su=5,sl=5,min_data=100,verbose=False)
                
                returning_spectra[value]=original_line/renormalisation_fit[0]
                returning_uncs[value]=uncertainty/renormalisation_fit[0]
            elif rgetattr(self,x+'.synth').any():
                
                original_line=rgetattr(self,x+'.spec_equalized')
                x_line=rgetattr(self,x+'.wave')
                synth_line=rgetattr(self,x+'.synth')
                uncertainty=rgetattr(self,x+'.uncs_equalized')
                
                renormalisation_fit = sclip((x_line,original_line/synth_line),chebyshev,int(3),ye=uncertainty,su=5,sl=5,min_data=100,verbose=False)
                
                # uncs_normal=poly_synth/poly_original*rgetattr(self,x+'.uncs')
                rsetattr(self,x+'.spec',original_line/renormalisation_fit[0])
                rsetattr(self,x+'.uncs',uncertainty/renormalisation_fit[0])
                self.limit_array()
            else:
                print(x+' Hasnt been synthesized')
            
        if not np.array_equal(data,None):
            return returning_spectra,returning_uncs
    def gradient(self,shift={},colours=None,self_normal=True,labels=['teff','logg','fe_h','Fe','alpha','vrad_Blue','vrad_Green','vrad_Red','vrad_IR','vsini','vmac','vmic']):
        if colours==None:
            colours=self.bands
        labels_payne=['teff','logg','fe_h','vmic','vsini','Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
        labels_payne_front=['teff','logg','fe_h','Fe','alpha']
        vrad_labels=['vrad_Blue','vrad_Green','vrad_Red','vrad_IR']
        labels_payne_front_wanted=[x for x in labels_payne_front if x in labels]
        labels_payne_wanted=[x for x in labels_payne if x in labels]

        rv_wanted=[x for x in vrad_labels if x in labels]
        synthetic_spectra,grad_all=self.synthesize(shift,give_back=True,grad=True)
        grad_out=np.ones(len(colours),dtype=object)
        if len(labels)-len(rv_wanted)!=0:
            for count,(syn,grad,x,current_label) in enumerate(zip(synthetic_spectra,grad_all,colours,labels_payne)):
                if self_normal:
                    grad_out[count]=-np.sum(np.array(grad)*rgetattr(self,x+'.limit')*(syn-rgetattr(self,x+'.spec'))/(rgetattr(self,x+'.uncs')**2),axis=1)/(rgetattr(self,x+'.x_max')-rgetattr(self,x+'.x_min'))
                else:
                    grad_out[count]=-np.sum(np.array(grad)*rgetattr(self,x+'.limit')*(syn-rgetattr(self,x+'.spec'))/(rgetattr(self,x+'.uncs')**2),axis=1)/(rgetattr(self,x+'.x_max')-rgetattr(self,x+'.x_min'))
            grad_out=np.sum(np.array(grad_out),axis=0)

        if len(rv_wanted)!=0:
            count=0
            grad_out_rv=np.ones(len(rv_wanted),dtype=object)
            for syn,x,current_labels in zip(synthetic_spectra,colours,vrad_labels):
                if current_labels in labels:
                    shift_temp=copy.copy(shift)
                    if 'vrad_'+x in shift_temp:
                        shift_temp['vrad_'+x]-=self.rv_shift
                    else:
                        shift_temp['vrad_'+x]=rgetattr(self,x+'.vrad')-self.rv_shift
                    grad_low=self.log_fit([x],shift_temp)
                    if 'vrad_'+x in shift_temp:
                        shift_temp['vrad_'+x]+=2*self.rv_shift
                    else:
                        shift_temp['vrad_'+x]=rgetattr(self,x+'.vrad')+self.rv_shift
                    grad_high=self.log_fit([x],shift_temp)
                    grad_out_rv[count]=(grad_high-grad_low)/(2*self.rv_shift)
                    count+=1
        if len(labels)-len(rv_wanted)!=0:
            grad_out_new=[x for x,y in zip(grad_out,labels_payne) if y in labels_payne_wanted]

            if len(rv_wanted)!=0:
                grad_out_new=np.array(np.hstack((grad_out_new[:len(labels_payne_front_wanted)],grad_out_rv,grad_out_new[len(labels_payne_front_wanted):])),dtype=np.float64)
        else:
            grad_out_new=grad_out_rv
        # print(grad_out_new)
        return np.array(grad_out_new,dtype=np.float64)
    
    def log_fit(self,colours=None,shift=None,dip_array=False,synthetic_spectra=None,normal=None,self_normal=False,solar_shift=None,uncertainty=None,limit_array=None,combine_masks=False):
        if colours==None:
            colours=self.bands
 
        limit_combine_array=[[0] for x in range(len(colours))]
        for value,x in enumerate(colours):
            if combine_masks:
                   if np.array_equal(limit_array,None):
                       limit_combine_array[value]=rgetattr(self,x+'.limit')*rgetattr(self,x+'.masked_area')
                   else:
                       limit_combine_array[value]=limit_array[value]*rgetattr(self,x+'.masked_area')
            else:
               if np.array_equal(limit_array,None):
                   limit_combine_array[value]=rgetattr(self,x+'.limit')
               else:
                   limit_combine_array[value]=limit_array[value]
        if not shift is None:
           synthetic_spectra=self.synthesize(shift,colours=colours,give_back=True)
        probability=np.ones(len(colours))
        if not dip_array:
           if np.array_equal(normal,None):
               for value,x in enumerate(colours):
                   if not(np.array_equal(synthetic_spectra,None)):
                       probability[value]= -np.sum((synthetic_spectra[value]-rgetattr(self,x+'.spec'))**2/(2*(rgetattr(self,x+'.uncs'))**2)*limit_combine_array[value])
                   else:
                       probability[value]= -np.sum((rgetattr(self,x+'.synth')-rgetattr(self,x+'.spec'))**2/(2*(rgetattr(self,x+'.uncs'))**2)*limit_combine_array[value])
           else:
               for value,x in enumerate(colours):
                   if not(np.array_equal(synthetic_spectra,None)):
                       probability[value]= -np.sum((synthetic_spectra[value]-normal[value])**2/(2*(uncertainty[value])**2)*limit_combine_array[value])
                   else:
                       probability[value]= -np.sum((rgetattr(self,x+'.synth')-normal[value])**2/(2*(uncertainty[value])**2)*limit_combine_array[value])
        else:
           if np.array_equal(normal,None):
               for value,x in enumerate(colours):
                   if not(np.array_equal(synthetic_spectra,None)):
                       probability[value]= -np.sum((synthetic_spectra[value]-rgetattr(self,x+'.spec'))**2/(2*(rgetattr(self,x+'.uncs'))**2)*limit_combine_array[value]*rgetattr(self,x+'.dip_array'))
                   else:
                       probability[value]= -np.sum((rgetattr(self,x+'.synth')-rgetattr(self,x+'.spec'))**2/(2*(rgetattr(self,x+'.uncs'))**2)*limit_combine_array[value]*rgetattr(self,x+'.dip_array'))
           else:
               for value,x in enumerate(colours):
                   if not(np.array_equal(synthetic_spectra,None)):
                       probability[value]= -np.sum((synthetic_spectra[value]-normal[value])**2/(2*(rgetattr(self,x+'.uncs'))**2)*limit_combine_array[value]*rgetattr(self,x+'.dip_array'))
                   else:
                       probability[value]= -np.sum((rgetattr(self,x+'.synth')-normal[value])**2/(2*(rgetattr(self,x+'.uncs'))**2)*limit_combine_array[value]*rgetattr(self,x+'.dip_array'))
        for value,x in enumerate(colours):
           if rgetattr(self,x+'.normal_value') is None:
               rsetattr(self, x+'.normal_value',probability[value]+1)
        if self_normal:
           normal_shift=np.ones(len(colours))
           for value,x in enumerate(colours):
               normal_shift[value]=rgetattr(self, x+'.normal_value')

           probability=probability-normal_shift
           if np.any(probability>0):
               for prob,x in zip(probability,colours):
                   if prob>0:
                       print(prob,x)
                       print('old',rgetattr(self, x+'.normal_value'))
                       rsetattr(self, x+'.normal_value',rgetattr(self, x+'.normal_value')+prob+1)
                       print('new',rgetattr(self, x+'.normal_value'))
                       print(solar_shift)
               return False      
        return sum(probability)
    def mass_setter(self,shift,colours=None):
        if colours==None:
            colours=self.bands

        for x in colours:
            for param in shift:
                if 'vrad' in param:
                    if x in param:
                        rsetattr(self,x+'.vrad',shift[param])
                else:
                    rsetattr(self,x+'.'+param,shift[param])

    def observed_spectra_giver(self,colours=None):
        if colours==None:
            colours=self.bands

        returning_spectra=np.array(np.ones(len(colours)),dtype=object)
        wavelengths=np.array(np.ones(len(colours)),dtype=object)
        uncs=np.array(np.ones(len(colours)),dtype=object)
        for value,x in enumerate(colours):
            returning_spectra[value]=rgetattr(self,x+'.spec')
            wavelengths[value]=rgetattr(self,x+'.wave')
            uncs[value]=rgetattr(self,x+'.uncs')
        return returning_spectra,wavelengths,uncs
    def plot(self,colours=None,lines='all',masks=False):
        if colours==None:
            colours=self.bands

        c=299792.458
        for x in colours:
            plt.figure()
            rv=rgetattr(self,x+'.vrad')
            x_line=rgetattr(self,x+'.wave')
            x_shifted=(1-rv/c)*x_line
            observed= runGetattr(self,x+'.spec')
            plt.plot(x_shifted, observed, label='Observed')

            # x_line=np.linspace(0,len(runGetattr(self,x+'.synth')[0])-1,num=len(runGetattr(self,x+'.synth')[0]))
            if rgetattr(self,x+'.synth').any():
                    # labels='synthetic  chi squared= '+str(self.log_fit([x]))
                    synthetic=runGetattr(self,x+'.synth')
                    plt.plot(x_shifted, runGetattr(self,x+'.synth'), label='Synthetic')
                    min_synth=min( runGetattr(self,x+'.synth'))
            if lines=='all':
                important_lines=rgetattr(self,x+'.important_lines')
                for individual_line in important_lines:
                    individual_line_temp=float(individual_line[1])
                    minimum=min(abs(x_line-individual_line_temp))
                    min_synth=min(observed[np.where(abs(x_line-individual_line_temp)==minimum)][0],synthetic[np.where(abs(x_line-individual_line_temp)==minimum)][0])
                    plt.axvline(x=individual_line_temp,c='pink')
                    plt.text(float(individual_line_temp),min_synth-0.05,individual_line[0][:2],fontsize=20,ha='center',color='pink')
            else:
                important_lines=rgetattr(self,x+'.important_lines')
                for individual_line in important_lines:
                    if individual_line[0][:2] in lines:
                        
                        plt.axvline(x=float(individual_line[1]))
                        plt.text(float(individual_line[1]),0.5,individual_line[0][:2],fontsize=20,ha='center',color='pink')
            if masks:
                vital_lines=rgetattr(self,x+'.vital_lines')
                for line in vital_lines:
                    plt.axvspan(line['line_begin'],line['line_end'],alpha=0.7,color='blue',label='vital lines')
                masks_line=rgetattr(self,x+'.masks')
                for line in masks_line:
                    plt.axvspan(line['mask_begin'],line['mask_end'],alpha=0.2,color='red',label='masks for bad spectra overall')
                # synthetic_spectra=rgetattr(self,x+'.synth')
                # observed_spectra=rgetattr(self,x+'.spec')
                # uncs=rgetattr(self,x+'.uncs')
                
                masks_bad_synth=rgetattr(self,x+'.masked_area')             
                limits_temp=[]
                limits_first_loop=[]
                first=True
                for wave_temp,mask_temp in zip(x_shifted,masks_bad_synth):
                    if not mask_temp:
                        if first:
                            limits_first_loop.append(wave_temp)
                            first=False
                    else:
                        if not first:
                            limits_first_loop.append(wave_temp)
                            limits_temp.append(limits_first_loop)
                            limits_first_loop=[]
                            first=True
                for area in limits_temp:
                    plt.axvspan(area[0],area[1],color='orange',alpha=0.2,label='bad  current synthetic spectra')
                limit_array=rgetattr(self,x+'.limit')
                limits_temp=[]
                limits_first_loop=[]
                first=True
                for wave_temp,mask_temp in zip(x_shifted,limit_array):
                    if not mask_temp:
                        if first:
                            limits_first_loop.append(wave_temp)
                            first=False
                    else:
                        if not first:
                            limits_first_loop.append(wave_temp)
                            limits_temp.append(limits_first_loop)
                            limits_first_loop=[]
                            first=True
                for area in limits_temp:
                    plt.axvspan(area[0],area[1],color='orange',alpha=0.2,label='limits')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.title(x+" Band")
            plt.xlim([x_shifted[0],x_shifted[-1]])
            plt.tight_layout()
def chebyshev(p,ye,mask):
    coef=np.polynomial.chebyshev.chebfit(p[0][mask], p[1][mask], 4)
    cont=np.polynomial.chebyshev.chebval(p[0],coef)
    return cont            
def starter_walkers_maker(nwalkers,old_abundances,parameters=['teff','logg','fe_h','fe_h','alpha','vrad_Blue','vrad_Green','vrad_Red','vrad_IR','vsini','vmac','vmic'],cluster=False):
    """
    Creates an 9 x nwalkers dimentional array thats is a good starter posistion for the walkers 

    Parameters
    ----------
    nwalkers : float 

    Returns
    -------
     the 9xn dimentional array 
    """
    pos=[]
    rv_labels={'vrad_Blue':0,'vrad_Green':1,'vrad_Red':2,'vrad_IR':3}
    parameters_Payne=['teff','logg','fe_h','vmic','vsini','Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
    elements=['Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
    if cluster:
        label_change={'teff':'teff_spectroscopic','logg':'logg_spectroscopic','vrad_Green':'red_rv_ccd'}
    else:
        label_change={'alpha':'alpha_fe_r','fe_h':'fe_h_r','vsini':'vbroad_r','vmic':'vmic_r','vmac':'vmic_r','vrad_Blue':'rv',
                           'vrad_Green':'rv','vrad_Red':'rv','vrad_IR':'rv','fe_h':'fe_h','teff':'teff_r','logg':'logg_r'}
    limits_Payne={x:[(x_max[x]+x_min[x])/2,(x_max[x]-x_min[x])/10] for x  in parameters if x in parameters_Payne}
    limits_rv={x:[0,0] for x  in parameters if not  x in parameters_Payne}
    # parameters_Payne=[x for x in parameters_Payne if x in parameters]
    # labels=['teff','logg','fe_h','fe_h','alpha','vsini','vmac','vmic']
    while np.shape(pos)[0]<nwalkers or len(np.shape(pos))==1:
        dictionary_parameters={}
        for value,x in enumerate(limits_Payne):
            if x in label_change:
                    labels=label_change[x]
            else:
                labels=x
            if x in elements:
                if cluster and isinstance(old_abundances[x.lower()+'_fe'],np.float32) and old_abundances[x.lower()+'_fe'] >x_min[x] and old_abundances[x.lower()+'_fe']<x_max[x]:
                    dictionary_parameters[x]=np.random.normal(old_abundances[x.lower()+'_fe'],old_abundances['e_'+x.lower()+'_fe'],1)
                else:
                    dictionary_parameters[x]=np.random.normal(0,0.1,1)
            elif old_abundances[labels] and not np.isnan( old_abundances[labels]) and old_abundances[labels]>x_min[x] and old_abundances[labels]<x_max[x]:
                dictionary_parameters[x]=abs(np.random.normal(old_abundances[labels],limits_Payne[x][1],1))
            else:
                dictionary_parameters[x]=np.random.normal(limits_Payne[x][0],limits_Payne[x][1],1)
        for x in limits_rv:
            if cluster:
                if not np.isnan( old_abundances['red_rv_ccd'][rv_labels[x]]):
                    dictionary_parameters[x]=np.random.normal(old_abundances['red_rv_ccd'][rv_labels[x]],1.0,1)
                elif not np.isnan(old_abundances['red_rv_ccd']):
                    dictionary_parameters[x]=np.random.normal(old_abundances['red_rv_ccd'],1.0,1)
    
                else:
                    dictionary_parameters[x]=np.random.normal(0,20.0,1)            
            else:
                if not np.isnan( old_abundances['rv'][rv_labels[x]]):
                    dictionary_parameters[x]=np.random.normal(old_abundances['rv'][rv_labels[x]],1.0,1)
                elif not np.isnan(old_abundances['rv_com']):
                    dictionary_parameters[x]=np.random.normal(old_abundances['rv_com'],1.0,1)
    
                else:
                    dictionary_parameters[x]=np.random.normal(0,20.0,1)            
        pos_try_number=np.hstack([dictionary_parameters[x] for x in parameters])
        if starting_test(pos_try_number,spectras.old_abundances,parameters=parameters,cluster=cluster):
            if len(pos)==0:
                pos=pos_try_number
            else:
                pos=np.vstack((pos_try_number,pos))
    return pos
def load_dr3_lines(mode_dr3_path = 'important_lines'):
    """
    
    """
    important_lines = [
        [4861.3230,r'H$_\beta$',r'H$_\beta$'],
        [6562.7970,r'H$_\alpha$',r'H$_\alpha$']
    ]
    
    important_molecules = [
        [4710,4740,'Mol. C2','Mol. C2'],
        [7594,7695,'Mol. O2 (tell.)','Mol. O2 (tell.)']
        ]

    line, wave = np.loadtxt(mode_dr3_path,usecols=(0,1),unpack=True,dtype=str, comments=';')

    for each_index in range(len(line)):
        if line[each_index] != 'Sp':
            if len(line[each_index]) < 5:
                important_lines.append([float(wave[each_index]), line[each_index], line[each_index]])
            else:
                important_lines.append([float(wave[each_index]), line[each_index][:-4], line[each_index]])
        
    return(important_lines,important_molecules)
def spread_masks(orginal_masks,spread=2):
    len_of_masks=len(orginal_masks)
    masks_temp=np.ones(len_of_masks,dtype=int)
    for value,y in enumerate(orginal_masks):
        if not y:
            for x in range(max(0,value-spread),min(len_of_masks,value+spread)):
                masks_temp[x]=0
    return masks_temp
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
colours_dict={'Blue':0,'Red':1,'Green':2,'IR':3}
labels=['teff','logg','fe_h','fe_h','alpha','vrad_Blue','vrad_Green','vrad_Red','vrad_IR','vsini','vmac','vmic']  



          


#Needs to be loaded now so the program works
global x_min,x_max
tmp = np.load("NN_normalized_spectra_all_elements_Blue.npz")   
labels_with_limits=['teff','logg','fe_h','vmic','vsini','Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']


x_min=tmp['x_min']
x_max=tmp['x_max']
x_min={x:y for x,y in zip(labels_with_limits,x_min)}
x_max={x:y for x,y in zip(labels_with_limits,x_max)}

# #EMCEE,
# prior=False
np.random.seed(589404)

parameters=['teff','logg','fe_h','vmic','vsini','vrad_Blue','vrad_Green','vrad_Red','vrad_IR','Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
parameters_no_elements=['teff','logg','fe_h','vmic','vsini','vrad_Blue','vrad_Green','vrad_Red','vrad_IR']
parameters_no_vrad=['teff','logg','fe_h','vmic','vsini','Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
elements=['Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
def main_analysis(sobject_id_name,ncpu=1):
    """
    main loop of the program, insert a sobject ID of the star you want to be reduced, insert also how many cpus you want it to run with

    Parameters
    ----------
    sobject_id_name : int
        sobject if of the star.
    ncpu : int, optional
        how many cpus do you want the program to run with . The default is 1.

    Returns
    -------
    None.

    """
    prior=False
    votable = parse("minimum_photometric_cross.xml")
    global photometric_data
    photometric_data=votable.get_first_table().to_table(use_names_over_ids=True)

    
    name=sobject_id_name
    # name=photometric_data[0]['sobject_id']
    global spectras
    spectras=spectrum_all(name,cluster=True)
    spectras.synthesize()
    spectras.normalize()
    global old_abundances
    old_abundances=spectras.old_abundances
    colours=spectras.bands
    reduction_status=np.any([rgetattr(spectras,x+'.bad_reduction') for x in colours ])or spectras.hermes_checker()
    
    if reduction_status:
          print('reduction failed will skip'+str(name)+ 'for now')
          return
    shift_radial={}
    print('calculating radial velocities')
    radial_velocities=[]
    for col in colours:
        logs=[]
        if not (np.isnan(old_abundances['red_rv_ccd'][colours_dict[col]]) or np.ma.is_masked(old_abundances['red_rv_ccd'][colours_dict[col]])):
            mean=float(old_abundances['red_rv_ccd'][colours_dict[col]])
        elif not( np.isnan(old_abundances['red_rv_com']) or np.ma.is_masked(old_abundances['red_rv_com'])):
            mean=float(old_abundances['red_rv_com'])
        else:
            mean=np.nanmean(photometric_data['red_rv_com'])
        if not (np.isnan(old_abundances['red_e_rv_ccd'][colours_dict[col]]) or np.ma.is_masked(old_abundances['red_e_rv_ccd'][colours_dict[col]])):
            sig=float(old_abundances['red_e_rv_ccd'][colours_dict[col]])*3
    
        elif not (np.isnan(old_abundances['red_e_rv_com']) or np.ma.is_masked(old_abundances['red_e_rv_com'])) :
            sig=float(old_abundances['red_e_rv_com'])*3
        else:
            sig=5
        num=int(np.ceil(min((sig*6)/0.1,30)))
        lin_vrad=np.linspace(mean-sig*3, mean+sig*3,num=num)
        lin_vrad_pool=[[x] for x in lin_vrad]
        # for x in lin_vrad_pool:
        #     logs.append(log_posterior(x,['vrad_'+col]))
        with Pool(processes=ncpu) as pool:
            logs=pool.map(partial(log_posterior,parameters=['vrad_'+col]),lin_vrad_pool)
        shift_radial['vrad_'+col]=lin_vrad[logs.index(max(logs))]
        radial_velocities.append(lin_vrad[logs.index(max(logs))])
    spectras.mass_setter(shift_radial)
    spectras.synthesize()
    spectras.normalize()
    pos_short=starter_walkers_maker(len(parameters_no_vrad)*2,old_abundances,parameters_no_vrad,cluster=True)
    ndim=np.shape(pos_short)[1]
    nwalkers=np.shape(pos_short)[0]
    
    step_iteration=20
    important_lines, important_molecules = load_dr3_lines()
    
    #This is the place where you can see the reduction in time. the EMCEE analysis calls the los_posterior to sample the probability space
    with Pool(processes=ncpu) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,pool=pool,args=[parameters_no_vrad,prior,parameters])
        
        print('doing first iteration for masks')
        #You can change the number of iterations to make it run for longer 
        for sample in sampler.sample(pos_short,iterations=10, progress=True):
            if sampler.iteration % step_iteration:
                    continue
# main_analysis(170506006401012)

#Uncomment this section of code if you would like to play around with it 
# votable = parse("minimum_photometric_cross.xml")
# photometric_data=votable.get_first_table().to_table(use_names_over_ids=True)

# spectras=spectrum_all(170830002301099)
# spectras.synthesize()
# spectras.normalize()
# old_abundances=spectras.old_abundances

#This creates some examples to run in the log posterior which is the main time sink of the process

#Just change 50 to how many examples you want
# pos_short=starter_walkers_maker(50,old_abundances,parameters_no_vrad,cluster=True)
# for pos in pos_short:
#     log_posterior(pos,parameters_no_vrad)
