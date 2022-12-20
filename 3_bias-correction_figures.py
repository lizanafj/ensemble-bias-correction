# -*- coding: utf-8 -*-

###############################################################################
# Program : 
# Authors  : Jesus Lizana
# Date	  : 1 July 2022
# Purpose : Bias correction 
##############################################################################


"""
BIAS CORRECTION APPROACH 
C.Figures before and after bias correction 

Inputs:
-Batches for historical/scenario 1.5/senario 2 (example: 923, 924 and 925)     - see line 38
-Same batches after bias correction (example: 923_bias, 924_bias and 925_bias)
-ERA5 file *.nc with same grid (example: ERA_5\ERA5_REGRID_ensemble_mean_2006-01_2016.12_6h-b3c29cc3e062.nc)

Outputs: 
-Generation of *.jpg file with data distribution before and after bias correction   - in folder: figures


"""       
  
#%%

##########################################################################
##########################################################################

#INPUT DATA - IMPORTANT TO CHECK BATCH Y YEARS ASSOCIATED

#Input of batch list for hist - 1.5 and 2 - in the same order
batch_list = [r"/batch_923",r"/batch_924",r"/batch_925"]

##########################################################################
##########################################################################

#%%

import os
import glob

print("....importing libraries")

import pandas as pd
import numpy as np

import netCDF4
from netCDF4 import Dataset,num2date # or from netCDF4 import Dataset as NetCDFFile

import xarray

#import shutil
import dateutil.parser

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.signal import detrend
import numpy.ma as ma
from statsmodels.distributions.empirical_distribution import StepFunction

import matplotlib.pyplot as plt    
import seaborn as sns

# get folder location of script
cwd = os.path.dirname(__file__) 
os.chdir(cwd)
#from bias_correction2 import XBiasCorrection, BiasCorrection #this is previous version, not used anymore

print("All libraries imported")


#%%

###########################################################

#Basic functions for netCDF4 files: 
    
###########################################################

def print_variables(data):
    """prints variables in netcdf Dataset
    only works if have assigned attributes!"""
    for i in data.variables:
        print(i, data.variables[i].units, data.variables[i].shape)
        
 
#%%

# get folder location of script
cwd = os.path.dirname(__file__) 

batch_hist = cwd+batch_list[0]
batch_15 = cwd+batch_list[1]
batch_2 = cwd+batch_list[2]

#%%

#Open batch_hist - list all files in list
os.chdir(batch_hist)
print("Opened: ",batch_hist)
batch_hist_files = glob.glob('item3236_6hrly_mean_*.nc')
batch_hist_files = sorted(batch_hist_files, key=str.lower)

#Open batch_hist - list all files in list
os.chdir(batch_15)
print("Opened: ",batch_15)
batch_15_files = glob.glob('item3236_6hrly_mean_*.nc')
batch_15_files = sorted(batch_15_files, key=str.lower)

#Open batch_hist - list all files in list
os.chdir(batch_2)
print("Opened: ",batch_2)
batch_2_files = glob.glob('item3236_6hrly_mean_*.nc')
batch_2_files = sorted(batch_2_files, key=str.lower)


#%%

print("")
print("Read ERA5")
print("")

os.chdir(batch_hist+"_temporal")
era = xarray.open_mfdataset("ERA_item3236_6hrly_mean_*.nc",combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,:,:]

era1 = xarray.concat([era,era,era],"new_dim")


"""
era1 = xarray.concat([era,era,era,era,era,era,era,era,era,era,
                  era,era,era,era,era,era,era,era,era,era,
                  era,era,era,era,era,era,era,era,era,era,
                  era,era,era,era,era,era,era,era,era,era,
                  era,era,era,era,era,era,era,era,era,era,
                  era,era,era,era,era,era,era,era,era,era,
                  era,era,era,era,era,era,era,era,era,era],"new_dim")

"""

era.close()
del era


#%%

######################################

#FIGURES WITH FINAL RESULTS BEFORE AND AFTER BIAS CORRECTION

########################################


print("starting figures")


os.chdir(batch_hist)
model = xarray.open_mfdataset(batch_hist_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]
print(model)

os.chdir(f"{batch_hist}_bias")
model_bias = xarray.open_mfdataset(batch_hist_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]
print(model_bias)


os.chdir(fr"{cwd}/figures")

era1_csv = era1[:,:,68,430].values-273.15
model_csv = model[:,:,68,430].values-273.15
model_bias_csv = model_bias[:,:,68,430].values-273.15


print("figure 1 - historical")


os.chdir(fr"{cwd}/figures")

#sns.distplot(era[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1, "label": "observed_ERA5_3",'linestyle':'--'})
sns.distplot(era1[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1.5, "label": "observed_ERA5",'linestyle':':'})
sns.distplot(model[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "r", "lw": 2, "label": "model_historical",'linestyle':'--'})
sns.distplot(model_bias[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "g", "lw": 2, "label": "model_historical_bias",'linestyle':':'})

plt.legend()
plt.xlabel("Temperature ºC")
#plt.xlim(-20, 50)
plt.title(f'Results after bias correction for {batch_list[0][1:]}')
plt.savefig(f"Fig_{batch_list[0][1:]}_bias.jpg",format="jpg")
plt.show()
plt.clf()

model.close()
model_bias.close()

del model
del model_bias

print("first figure done")
             
   
#%%  

print("figure 2 - scenario 1.5")

os.chdir(batch_15)
model = xarray.open_mfdataset(batch_15_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]

os.chdir(f"{batch_15}_bias")
model_bias = xarray.open_mfdataset(batch_15_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]


os.chdir(fr"{cwd}/figures")

#sns.distplot(era[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1, "label": "observed_ERA5_3",'linestyle':'--'})
sns.distplot(era1[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1.5, "label": "observed_ERA5",'linestyle':':'})
sns.distplot(model[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "r", "lw": 2, "label": "model_scenario1.5",'linestyle':'--'})
sns.distplot(model_bias[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "g", "lw": 2, "label": "model_scenario1.5_bias",'linestyle':':'})

plt.legend()
plt.xlabel("Temperature ºC")
#plt.xlim(-20, 50)
plt.title(f'Results after bias correction for {batch_list[1][1:]}')
plt.savefig(f"Fig_{batch_list[1][1:]}_bias.jpg",format="jpg")
plt.show()
plt.clf()

model.close()
del model
model_bias.close()
del model_bias

print("second figure done")
              
      
#%%

print("figure 3 - scenario 2")

os.chdir(batch_2)
model = xarray.open_mfdataset(batch_2_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]

os.chdir(f"{batch_2}_bias")
model_bias = xarray.open_mfdataset(batch_2_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]

os.chdir(fr"{cwd}/figures")

#sns.distplot(era[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1, "label": "observed_ERA5_3",'linestyle':'--'})
sns.distplot(era1[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1.5, "label": "observed_ERA5",'linestyle':':'})
sns.distplot(model[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "r", "lw": 2, "label": "model_scenario2",'linestyle':'--'})
sns.distplot(model_bias[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "g", "lw": 2, "label": "model_scenario2_bias",'linestyle':':'})

plt.legend()
plt.xlabel("Temperature ºC")
#plt.xlim(-20, 50)
plt.title(f'Results after bias correction for {batch_list[2][1:]}')
plt.savefig(f"Fig_{batch_list[2][1:]}_bias.jpg",format="jpg")
plt.show()
plt.clf()


model.close()
del model
model_bias.close()
del model_bias

                    
#%%









