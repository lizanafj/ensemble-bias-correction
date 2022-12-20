# -*- coding: utf-8 -*-

###############################################################################
# Program : 
# Authors  : Jesus Lizana
# Date	  : 1 July 2022
# Purpose : Bias correction 
##############################################################################


"""
BIAS CORRECTION APPROACH 
A.PRE-PROCESSING

Inputs:
-Select batches for historical/scenario 1.5/senario 2 (example: 923, 924 and 925)    - see line 38
-ERA5 file *.nc with same grid (example: ERA_5\ERA5_REGRID_ensemble_mean_2006-01_2016.12_6h-b3c29cc3e062.nc)


Outputs: 
-Creation of folders structure (batch_bias, temporal, figures)
-Calculation of data distribution - identification of ourtliers
-Generation of *.csv file per scenario - for one specific lat/long - to identify outliers

Note: 
Update step 2 before starting - index of observed and model should be the same


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

import numpy as np
import pandas as pd

import netCDF4
from netCDF4 import Dataset,num2date # or from netCDF4 import Dataset as NetCDFFile

import xarray

#import shutil
import dateutil.parser

import matplotlib.pyplot as plt    
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.signal import detrend
import numpy.ma as ma
from statsmodels.distributions.empirical_distribution import StepFunction

# get folder location of script
cwd = os.path.dirname(__file__) 
os.chdir(cwd)

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


ERA5_file_REGRID= cwd+"/ERA_5/ERA5_REGRID_ensemble_mean_2006-01_2016.12_6h-b3c29cc3e062.nc"
#ERA5_file= cwd+"\ERA_5\ERA5_ensemble_mean_2006-01_2016.12_6h-b3c29cc3e062.nc"


#%%

#Check if folders for bias created - if not - create
if not os.path.exists(batch_hist+"_bias"):
    os.makedirs(batch_hist+"_bias")
    print("new folder created: ", batch_hist+"_bias")
    
if not os.path.exists(batch_hist+"_temporal"):
    os.makedirs(batch_hist+"_temporal")
    print("new folder created: ", batch_hist+"_temporal")
    
    
if not os.path.exists(batch_15+"_bias"):
    os.makedirs(batch_15+"_bias")
    print("new folder created: ", batch_15+"_bias")
    
if not os.path.exists(batch_2+"_bias"):
    os.makedirs(batch_2+"_bias")
    print("new folder created: ", batch_2+"_bias")
    
if not os.path.exists(fr"{cwd}/figures"):
    os.makedirs(fr"{cwd}/figures")
    print("new folder created: ", fr"{cwd}/figures")


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

####################################################

##STEP 1- CREATE ERA5 members for the same period

####################################################

print("")
print("CREATING ERA5 members for the timeframe for bias correction ")
print("")
        
#List of files of different years
yearrange = []
for i in batch_hist_files:
    year_start = i[25:40]
    yearrange.append(year_start)
yearrange = list(set(yearrange))
yearrange.sort()

  
#Selection of one file per year to get index 
files_forERA = []
for n in yearrange: 
    str_match = [s for s in batch_hist_files if str(n) in s]
    if len(str_match)>0:
        files_forERA.append(str_match[0])
        

print("Files used as a reference: ", len(files_forERA))
print(files_forERA)
                     
#%%

#Go to main directory: 
os.chdir(cwd)
print("Opened: ",cwd)
  

for i in files_forERA:
    
    #create new file win temporal folder
    src= f"{batch_hist}/{i}"
    #dst = f"{batch_hist}_temporal\ERA_{i}"
    #shutil.copyfile(src, dst)    
            
    print("")
    print("Step 2 - Extracting from ERA5...... - Reference file: ", i)
    print("")
    
    
    #Get starting date and ending date of mean file ###########################################################
    print("Period: ")
    data = Dataset(src, mode='r', format="NetCDF") #dir_path + 

    #Print starting date (alternative aproach)
    times = data.variables['time0']
    dates = num2date(times[:],times.units, times.calendar)
    print('starting date = %s' % dates[0])
    print('ending date = %s'% dates[-1])

    #create datetime from num to date
    tname = "time0"
    nctime = data.variables[tname][:] # get values
    t_unit = data.variables[tname].units # get unit  "days since 1950-01-01T00:00:00Z"
    t_cal = data.variables[tname].calendar
    tvalue = num2date(nctime,units = t_unit,calendar = t_cal)
    str_time = [i.strftime("%Y-%m-%d %H:%M") for i in tvalue] # to display dates as string

    start = str_time[0]
    end = str_time[-1]

    #select period in datetime
    sdt = dateutil.parser.parse(start)
    edt = dateutil.parser.parse(end)

    print ("selected period :", sdt, edt)
    longitud = len(times[:])

    data.close()

    
    #Extraction ERA5 ###########################################################
         
    #Open ERA5 and get selected variables
    data = Dataset(ERA5_file_REGRID, "r", format = "NETCDF4")
    
    print("")
    print("Variables ERA")
    print_variables(data)
   

    #check datatime in new nc file updated
    times = data.variables['time0']
    dates = num2date(times[:],times.units,times.calendar)
    print('starting date = %s' % dates[0])
    print('ending date = %s'% dates[-1])

          
    ##Identify selected period in nc index of ERA5 - convert from date to index netcd4f: 
    t_unit = times.units # get unit  "days since 1950-01-01T00:00:00Z"
    t_cal = times.calendar
    st_idx = netCDF4.date2index(sdt, times,calendar = t_cal)
    et_idx = netCDF4.date2index(edt, times,calendar = t_cal)

    print("Index values: ", st_idx, et_idx, et_idx-st_idx)
    
    #Correction of index to avoid assumptions between ERA and simulations (años bisiestos, 30 days per month, etc)
    et_idx = st_idx+longitud-1
    print("index corrected: ", st_idx, et_idx, et_idx-st_idx)

      
    ##### create dimensions for new file ####
    input_lat_dim = data.dimensions["latitude0"]
    input_lon_dim = data.dimensions["longitude0"]
    input_time_dim = data.dimensions["time0"]
             
    ##### read variables ####
    lat = data.variables['latitude0']
    lon = data.variables['longitude0']
    time = data.variables['time0'] #[st_idx:et_idx+1]
    time0 = data.variables['time0'][st_idx:et_idx+1]
   
    temp = data.variables["item3236_6hrly_mean"] #[st_idx:et_idx+1, :, :]
    temp0 = data.variables["item3236_6hrly_mean"][st_idx:et_idx+1, :, :]

    
    #Open ERA5 and get selected variables
    
    #data from i
    data_mean = Dataset(src, "r", format = "NETCDF4")
    time2 = data_mean.variables['time0']
 
    #CREATION OF NEW ERA5 WITH SELECTED PERIOD AND VARIABLES FROM RAW ERA5
    New_ERA = f"{batch_hist}_temporal/ERA_{i}"

    #### open filestreams and create new filestream for pvout ######
    output_stream = Dataset(New_ERA, "w", format = "NETCDF4")

    ##### create dimensions for new file ####
    output_stream.createDimension("latitude0", len(input_lat_dim))
    output_stream.createDimension("longitude0", len(input_lon_dim))
    output_stream.createDimension("time0", None)

    ##### create variables ####
    output_lat_var = output_stream.createVariable("latitude0",lat.datatype,("latitude0",))
    output_lat_var.units = 'degrees_north'

    output_lon_var = output_stream.createVariable("longitude0",lon.datatype,("longitude0",))
    output_lon_var.units = 'degrees_east'

    output_time_var = output_stream.createVariable("time0",time2.datatype,("time0",))
    output_time_var.units = 'days since 2005-12-01 00:00:00' #'hours since 1900-01-01 00:00:00.0'
    output_time_var.calendar = time2.calendar


    output_temp_var = output_stream.createVariable("item3236_6hrly_mean",temp.datatype,("time0","latitude0","longitude0"))
    output_temp_var.units = 'K'

    #### writing data ###
    output_lat_var[:] = lat[:]
    output_lon_var[:] = lon[:]
    output_time_var[:] = time2[:]
    output_temp_var[:] = temp0

    print("")
    print("Variables new ERA for bias correction")
    print_variables(output_stream)
       

    ###### close filestreams #####
    output_stream.close()
    data_mean.close()
    
    print("")
    print(f"ERA_{i}", " - File ERA created with same period")

 
print("")
print("ALL ERA-based FILES CREATED FOR THE SAME BATCH PERIOD")      

                 
#%%  

####################################################

##STEP 2 - READ ERA5 for the same timeframe

####################################################

print("")
print("READ ERA5 for the same timeframe")
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


####################################################

## STEP 3 - Pre-processing - identification of outliers

####################################################

os.chdir(cwd)
print("star figures")


os.chdir(batch_hist)
model = xarray.open_mfdataset(batch_hist_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]

os.chdir(fr"{cwd}/figures")

#sns.distplot(era[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1, "label": "observed_ERA5_3",'linestyle':'--'})
sns.distplot(era1[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1.5, "label": "observed_ERA5",'linestyle':':'})
sns.distplot(model[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "r", "lw": 2, "label": "model_historical",'linestyle':'--'})
#sns.distplot(model_bias[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "g", "lw": 2, "label": "model_historical_bias",'linestyle':':'})

plt.legend()
plt.xlabel("Temperature ºC")
#plt.xlim(-20, 50)
plt.title(f'Data distribution for {batch_list[0][1:]}')
plt.savefig(f"Fig_{batch_list[0][1:]}_bias.jpg",format="jpg")
plt.show()
plt.clf()


model.close()
#model_bias.close()
del model
#del model_bias

print("first figure done")
                
#%%  


os.chdir(batch_15)
model = xarray.open_mfdataset(batch_15_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]

#os.chdir(f"{batch_15}_bias")
#model_bias = xarray.open_mfdataset(batch_15_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]


os.chdir(fr"{cwd}/figures")

#sns.distplot(era[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1, "label": "observed_ERA5_3",'linestyle':'--'})
sns.distplot(era1[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1.5, "label": "observed_ERA5",'linestyle':':'})
sns.distplot(model[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "r", "lw": 2, "label": "model_scenario1.5",'linestyle':'--'})
#sns.distplot(model_bias[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "g", "lw": 2, "label": "model_scenario1.5_bias",'linestyle':':'})

plt.legend()
plt.xlabel("Temperature ºC")
#plt.xlim(-20, 50)
plt.title(f'Data distribution for {batch_list[1][1:]}')
plt.savefig(f"Fig_{batch_list[1][1:]}_bias.jpg",format="jpg")

plt.show()
plt.clf()

model.close()
del model
#model_bias.close()
#del model_bias

print("second figure done")
                    
#%%

os.chdir(batch_2)
model = xarray.open_mfdataset(batch_2_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]

#os.chdir(f"{batch_2}_bias")
#model_bias = xarray.open_mfdataset(batch_2_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]



os.chdir(fr"{cwd}/figures")

#sns.distplot(era[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1, "label": "observed_ERA5_3",'linestyle':'--'})
sns.distplot(era1[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "b", "lw": 1.5, "label": "observed_ERA5",'linestyle':':'})
sns.distplot(model[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "r", "lw": 2, "label": "model_scenario2",'linestyle':'--'})
#sns.distplot(model_bias[:,:,68,430].values-273.15, hist=False, kde_kws={"color": "g", "lw": 2, "label": "model_scenario2_bias",'linestyle':':'})

plt.legend()
plt.xlabel("Temperature ºC")
#plt.xlim(-20, 50)
plt.title(f'Data distribution for {batch_list[2][1:]}')
plt.savefig(f"Fig_{batch_list[2][1:]}_bias.jpg",format="jpg")

plt.show()
plt.clf()

model.close()
del model
#model_bias.close()
#del model_bias


                
#%% 

####################################################

## STEP 4 - Save CSV files to identify wrong files: 

####################################################

#ERA5
os.chdir(fr"{cwd}/figures")
era1_csv = era1[:,:,68,430].values-273.15
np.savetxt("era5.csv",era1_csv,delimiter=",")


#HISTORICAL
os.chdir(batch_hist)
model = xarray.open_mfdataset(batch_hist_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]
print(model)

os.chdir(fr"{cwd}/figures")
model_csv = model[:,:,68,430].values-273.15
np.savetxt("model_historical.csv",model_csv,delimiter=",")

del model
del model_csv


#SCENARIO 1.5
os.chdir(batch_15)
model = xarray.open_mfdataset(batch_15_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]
model_csv = model[:,:,68,430].values-273.15
os.chdir(fr"{cwd}/figures")
np.savetxt("model_scenario15.csv",model_csv,delimiter=",")

del model
del model_csv


#SCENARIO 2
os.chdir(batch_2)
model = xarray.open_mfdataset(batch_2_files,combine = "nested",concat_dim="new_dim",decode_times=False)['item3236_6hrly_mean'][:,:,0,:,:]
model_csv = model[:,:,68,430].values-273.15
os.chdir(fr"{cwd}/figures")
np.savetxt("model_scenario2.csv",model_csv,delimiter=",")

                
#%%  

#List of files per folder
df_1 = pd.DataFrame(batch_hist_files, columns=["files"])
df_1.to_csv('batch_hist_files.csv', index=False)

df_2 = pd.DataFrame(batch_15_files, columns=["files"])
df_2.to_csv('batch_15_files.csv', index=False)

df_3 = pd.DataFrame(batch_2_files, columns=["files"])
df_3.to_csv('batch_2_files.csv', index=False)

                    
#%%
