## coding: utf-8 
#!/usr/bin/env python2/3
#
# File: main_method_comp_script.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Main Script that will calculate distances to stars (using the absolute magnitudes and the PML relations), convert these to parallaxes and perform the actual method comparison

import numpy as np
import pandas as pd

from uncertainties import unumpy

import re
import inspect

import sys

import PML_relations as PML

import Distance_Parallax as Dplx

import Tukey_Bland_Altman_Krouwer as BA

import Passing_Bablok as PB

from astropy.io import ascii

import matplotlib.pyplot as plt

###############################################################################
#                               Setup for Script                              #
###############################################################################

#------------------------------------DATA INPUT--------------------------------
# Indicate whether or not to use data from our dereddening script! (If True, you have to provide these down below!)
our_script = False

# INDICATE HOW YOU WOULD LIKE TO PROVIDE DATA INPUT (one of these should be set to True)
using_create_data = False # THIS IS SEMI-MANUAL: create_data.py queries the GAIA parallaxes and W1, Ks magnitudes, other data needs to be provided manually (Period,Metallicity,BRR,RRABC,Vmag,eVmag)
full_manual_input = False # FULL MANUAL INPUT
using_csv_input = True # USING A USER-MADE CSV FILE: user-specified format, might be useful for bigger datasets

# Specify whether or not a (not really correct) assumed error (the precision of the R-factor itself) on the R-factor
# in the dereddening analysis should be used for dust map dereddening
R_error = False
#------------------------------------DATA INPUT--------------------------------


#-----------------------------------DATA OUTPUT--------------------------------
# Select which relations need to be compared: (set second relation to '' if you only need one, in order to compare to GAIA parallaxes!)
first_relation = 'Sesar '
second_relation = ''
# SELECT FROM THE FOLLOWING LIST: 'Dambis V','Dambis Ks','Dambis W1','Klein Mv','Klein W1','Muraveva LMC','Muraveva plx','Neeley ','Sesar '

# INDICATE WHAT YOU WOULD LIKE FOR METHOD COMPARISON OUTPUT: Passing-Bablok regression and/or Bland-Altman plots
BLANDALTMAN = False # Bland-Altman or Tukey's mean difference and Krouwer plots
outfile_BA = 'BA_results.txt' # specify name of file containing the printed outputs of Tukey_Bland_Altman_Krouwer.py
PASSINGBABLOK = True # Passing Bablok regression plots
outfile_PB = 'PB_results.txt' # specify name of file containing the printed outputs of Passing_Bablok.py

# indicate whether or not absolute values will be displayed on the Bland-Altman or Tukey's mean difference and Krouwer plots
# IF BOTH percent_BA and log_BA = FALSE --> use absolute values; OTHERWISE set one of these to True! (Our personal experience: not recommended for this validation...)
percent_BA = False # plot the Bland-Altman or Tukey's mean difference and Krouwer plots in percentages (with propagated uncertainties)
log_BA = False # Bland-Altman or Tukey's mean difference and Krouwer plots in logscale (with propagated uncertainties)

# Indicate whether or not you would like to (only) print the distances for the stars using the set of PML relations
distanceprint = False
# Indicate whether or not you would like to (only) print the parallaxes for the stars using the set of PML relations
parallaxprint = False
# Indicate whether or not you would like to (only) print the stars for which negative GAIA parallaxes were found
print_neg_GAIA = False
# Indicate whether or not you would like to (only) print the RRtype and Blazhko variability statistics
print_BRR_RRAB = False
#-----------------------------------DATA OUTPUT--------------------------------


#----------------------------------DUST TABLES---------------------------------
# set the directory of the txt-file containing your output from NASA/IPAC  INFRARED  SCIENCE  ARCHIVE Galactic Dust Reddening and Extinction 'query'
Directory_EBV = '/Users/jvb/Documents/Proposal2018/Input_creation/'
# filename of the txt-file containing output from NASA/IPAC  INFRARED  SCIENCE  ARCHIVE Galactic Dust Reddening and Extinction 'query'
filename_EBV = 'Dustmap_output_table.txt'
#----------------------------------DUST TABLES---------------------------------

#################### FUNDAMENTAL PARAMETERS / OBSERVABLES ######################

# standardized input for the main script, which will be overwritten when using csv input

#------------------------------------PERIOD------------------------------------
# specify RR Lyrae pulsation period (overwritten in csv option)
Per = np.array([0.7100,0.4800,0.4400,0.3900,0.6500,0.5000,0.3100,0.4200]) # if error is estimated to be 10-4 
e_Per = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]) # estimated to be 10-4
#------------------------------------PERIOD------------------------------------

#----------------------------------METALLICITY---------------------------------
# specify determined metallicity for RR Lyrae stars in sample (overwritten in csv option)
FE = np.array([-1.83,-0.57,-0.04,-1.65,-2.5,1.5,-1.77,1.5]) # --> first estimates from Simbad/Vizier!! HAVE TO CHANGE
e_FE = np.array([0.5,0.5,0.5,0.11,0.5,0.5,0.5,0.5]) 
#----------------------------------METALLICITY---------------------------------

#-----------------------------------BLAZKHO------------------------------------
# specify whether or not RR Lyrae stars in sample display Blazhko effect (overwritten in csv option)
BRR = np.array(["Blazhko","Blazhko","Blazhko","Blazhko","RRLyr","RRLyr","RRLyr","RRLyr"])
#-----------------------------------BLAZKHO------------------------------------

#-------------------------------------RRABC------------------------------------
# specify type of RR Lyrae variability for all the stars in the sample (overwritten in csv option)
RRABC = np.array(["RRAB","RRAB","RRAB","RRC","RRAB","RRAB","RRC","RRC"])
#-------------------------------------RRABC------------------------------------

#-------------------------------------VMAG-------------------------------------
# specify visual magnitude of RR Lyrae sample (overwritten in csv option)
Vmag = np.array([9.76,9.86,9.48,10.095,9.395,9.3,9.322,10.2665]) #visual observed magnitudes --> TAKING MIDDLE OF RANGE
eVmag = np.array([0.2,0.46,0.34,0.165,0.425,0.2,0.212,0.0335]) # TAKING SYMMETRIC ERROR AROUND MIDDLE
#-------------------------------------VMAG-------------------------------------
 
#################### FUNDAMENTAL PARAMETERS / OBSERVABLES ######################

###############################################################################   
# define the star names (overwritten in the csv load option)
stars = ['SV Eri','XZ Dra','SW And','RU Psc', 'X Ari','BD+184995','RZ Cep','V1057 Cas']

# rows of the pandas dataframe that contains the stars, their known periods, their derived metallicities, RRAB/RRC (RRab,RRc type), B/RR (Blazhko/non-Blazhko)
rows = ['P','e_P','Fe','e_Fe','B/RR','RRAB/RRC','Vmag','e_Vmag','Ksmag','e_Ksmag','W1mag','e_W1mag']
###############################################################################   

#------------------------------Reddening---------------------------------------
    
# read in the (IPAC-table format) txt-file containing interstellar reddening information from DUST TABLES
EBV_data = ascii.read(Directory_EBV+filename_EBV)
# convert to dataframe, name columns (stars)
EBV_df = EBV_data.to_pandas().T
EBV_df.columns = stars
# rename certain table/dataframe rows to readable rows for the main program, so that your dust map reddening information is ready to be processed!
EBV_df = EBV_df.rename(index={"mean_E_B_V_SandF": "EBV_S_F", "stdev_E_B_V_SandF": "e_EBV_S_F","mean_E_B_V_SFD":"EBV_SFD","stdev_E_B_V_SFD":"e_EBV_SFD"},columns=str)

#------------------------------------------------------------------------------
    
# using OUR DEREDDENING SCRIPT (dereddening.py): i.e. 'manual' dereddening via Monte Carlo analysis
VEXT = np.array([0.264,0.037,0.166,0.343,0.52,0.30,0.66,0.68]) #visual extinction
eVEXT = np.array([0.002,0.001,0.002,0.001,0.01,0.01,0.02,0.01])

W1EXT = np.array([0.0173,0.0025,0.0109,0.0225,0.034,0.020,0.043,0.045]) #W1 extinction
eW1EXT = np.array([0.0002,0.0001,0.0002,0.0001,0.001,0.002,0.003,0.001])
    
KsEXT = np.array([0.0305,0.0044,0.0193,0.0397,0.060,0.035,0.075,0.079]) #Ks extinction
eKsEXT = np.array([0.0003,0.0002,0.0003,0.0002,0.002,0.003,0.005,0.002])    
    
# naming the rows of the df
rows_script_ext = ['V','e_V','Ks','e_Ks','W1','e_W1']
# create dataframe containing the visual extinctions according to the three methods used
script_ext_df = pd.DataFrame(np.vstack((VEXT,eVEXT,KsEXT,eKsEXT,W1EXT,eW1EXT)), index=rows_script_ext, columns=stars)

#------------------------------Reddening--------------------------------------- 


############################FULL MANUAL INPUT##################################

if full_manual_input:
    
    # specify observed magnitudes (2MASS Ks, ALLWISE W1)
    Ksmag = np.array([8.572,9.147,8.444,9.068,7.847,8.461,7.967,8.534]) #Ks observed magnitudes
    eKsmag = np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]) # taking B.C. as error --> not correct
    W1mag = np.array([8.519,9.081,8.471,9.058,7.866,8.418,7.846,8.282]) #W1 observed magnitudes
    eW1mag = np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]) # taking B.C. as error --> not correct
    
    # construct a dataframe encompassing all that information
    df = pd.DataFrame(np.vstack((Per,e_Per,FE,e_FE,BRR,RRABC,Vmag,eVmag,Ksmag,eKsmag,W1mag,eW1mag)), index=rows, columns=stars)
    
    #---------------------------------------------------------------------------------------
    
    # specify GAIA parallaxes (in mas)
    GAIAparallaxes = [1.217,1.296,1.78,0.911,1.836,2.259,2.359,2.167]
    e_GAIAparallaxes = [0.058,0.024,0.16,0.077,0.041,0.073,0.029,0.027]
    uGAIAparallaxes = unumpy.uarray(GAIAparallaxes,e_GAIAparallaxes)
    
    #---------------------------------------------------------------------------------------
    
    # Select a subset of all parameters in the original dataframe, containing the observed magnitudes
    ms_df = df.copy().loc[['Vmag','e_Vmag','Ksmag','e_Ksmag','W1mag','e_W1mag']]
    # Make a pandas dataframe containing the ufloats (float64) of the observed magnitudes in different passbands
    m_df = pd.DataFrame([unumpy.uarray(ms_df.iloc[range(0, len(ms_df.values), 2)].iloc[j].astype(np.float64).tolist()
    ,ms_df.iloc[range(1, len(ms_df.values), 2)].iloc[h].astype(np.float64).tolist()) for j,h in zip(range(len(ms_df.values)/2),range(len(ms_df.values)/2))]
    , index=['V','Ks','W1'],columns=stars)
    # Make sure df columns are sorted in correct way
    m_df = m_df.reindex(columns=stars)

############################FULL MANUAL INPUT##################################


############################USING CREATE DATA##################################
if using_create_data:
    
    # Load the data from the .dat file created by create_data.py
    create_output = np.genfromtxt('W_K_plx.dat',delimiter='\t',names=True)
    
    # specify observed magnitudes (2MASS Ks, ALLWISE W1)
    Ksmag = create_output['KMag'] #Ks observed magnitudes
    eKsmag = create_output['eKMag'] 
    W1mag = create_output['W1Mag'] #W1 observed magnitudes
    eW1mag = create_output['eW1Mag'] 
    
    # construct a dataframe encompassing all that information
    df = pd.DataFrame(np.vstack((Per,e_Per,FE,e_FE,BRR,RRABC,Vmag,eVmag,Ksmag,eKsmag,W1mag,eW1mag)), index=rows, columns=stars)
    
    #---------------------------------------------------------------------------------------
    
    # specify GAIA parallaxes (in mas)
    GAIAparallaxes = create_output['PLX']
    e_GAIAparallaxes = create_output['ePLX']
    uGAIAparallaxes = unumpy.uarray(GAIAparallaxes,e_GAIAparallaxes)
    
    #---------------------------------------------------------------------------------------
    
    # Select a subset of all parameters in the original dataframe, containing the observed magnitudes
    ms_df = df.copy().loc[['Vmag','e_Vmag','Ksmag','e_Ksmag','W1mag','e_W1mag']]
    # Make a pandas dataframe containing the ufloats (float64) of the observed magnitudes in different passbands
    m_df = pd.DataFrame([unumpy.uarray(ms_df.iloc[range(0, len(ms_df.values), 2)].iloc[j].astype(np.float64).tolist()
    ,ms_df.iloc[range(1, len(ms_df.values), 2)].iloc[h].astype(np.float64).tolist()) for j,h in zip(range(len(ms_df.values)/2),range(len(ms_df.values)/2))]
    , index=['V','Ks','W1'],columns=stars)
    # Make sure df columns are sorted in correct way
    m_df = m_df.reindex(columns=stars)

############################USING CREATE DATA##################################
    
    
#############################USING CSV INPUT###################################

if using_csv_input:
    
    csv_path = '/Users/jvb/Documents/Proposal2018/GAIA/'
    csv_filename = "RRL_Dambis2013.csv"
    csv_path_blazhko = '/Users/jvb/Documents/Proposal2018/PassingBland/'
    csv_filename_blazhko = "BLAZHKO_csv.dat"

    data_csv = pd.read_csv(csv_path+csv_filename)
    
    # specify fundamental parameters/information on the RR LYRAE stars in our sample
    Per = data_csv["Period"].values # periods
    e_Per = np.zeros_like(Per) # errorless periods
    FE = data_csv["FeH"].values # metallicity (Fe/H)
    e_FE = np.zeros_like(FE) # errorless metallicities
    BRR = ["Blazhko"]*len(FE) # dummy BRR column that will be replaced later
    RRABC = data_csv["RRtype"].replace({'AB': 'RRAB','C': 'RRC'}).values
    
    # specify observed magnitudes (Visual, 2MASS Ks, ALLWISE W1)
    Vmag = data_csv["Vmag"].values # visual observed magnitudes 
    eVmag = data_csv["e_Vmag"].values 
    Ksmag = data_csv["Kmag"].values # Ks observed magnitudes
    eKsmag = data_csv["e_Kmag"].values 
    W1mag = data_csv["W1mag"].values # W1 observed magnitudes
    eW1mag = data_csv["e_W1mag"].values 
    
    # construct a dataframe encompassing all that information
    non_unique_df = pd.DataFrame(np.vstack((Per,e_Per,FE,e_FE,BRR,RRABC,Vmag,eVmag,Ksmag,eKsmag,W1mag,eW1mag)), index=rows, columns=data_csv['Name'].values)
    NANs_list = non_unique_df.isnull().stack()[lambda x: x].index.tolist() # find tuples of positions where NaN values occur in the dataframe
    unique_Nan_star_list = list(set([y[1] for y in NANs_list])) # generate the unique list of stars for which the NaN values occur
    df = non_unique_df.drop(unique_Nan_star_list, axis=1) # drop these stars from the dataframe
       
    #---------------------------------------------------------------------------------------

    csv_path_blazhko = '/Users/jvb/Documents/Proposal2018/PassingBland/'
    csv_filename_blazhko = "BLAZHKO_csv.dat"
    blazhko_data_csv = np.genfromtxt(csv_path_blazhko+csv_filename_blazhko,delimiter='\t',dtype=['S20','S20'],names=True)
    BRR = blazhko_data_csv["Blazhko"]

    # use the stars for which a blazhko type (i.e. either RRLyr or Blazhko) has been found
    common_cols_blazhko = df.columns.intersection(blazhko_data_csv["Starname"])

    # get indices of common columns in dataframe containing unique stars without NaNs
    common_col_indices_blazhko = [df.columns.get_loc(i) for i in common_cols_blazhko]

    # slice the dataframe based on the common columns
    df = df[common_cols_blazhko]    

    # overwrite stars, RRABC, BRR (only selecting useful columns)
    stars = common_cols_blazhko.values
    BRR = df.loc[["B/RR","RRAB/RRC"]].values[0]
    RRABC = df.loc[["B/RR","RRAB/RRC"]].values[1]    
 
    # overwrite the BRR column of the dataframe, containing the correct Blazhko type
    df.loc["B/RR"] = blazhko_data_csv["Blazhko"]    
    BRR = blazhko_data_csv["Blazhko"]

    #---------------------------------------------------------------------------------------

    # Load the data from the .dat file created by create_GAIA_data.py
    create_GAIA_output = np.genfromtxt('GAIA_DATA_csv.dat',delimiter='\t',names=True,dtype=['S20','f8','f8'])
    
    # obtain slicing indices for the GAIA data based on whether or not a Blazhko type is identified
    ind_dict = dict((k,i) for i,k in enumerate(create_GAIA_output['Starname']))
    inter = set(ind_dict).intersection(common_cols_blazhko)
    indices_blazhko_GAIA = [ ind_dict[x] for x in inter ]       
    
    # specify GAIA parallaxes (in mas)
    GAIAparallaxes = create_GAIA_output["Parallax"]
    e_GAIAparallaxes = create_GAIA_output["e_Parallax"]
    uGAIAparallaxes = unumpy.uarray(GAIAparallaxes,e_GAIAparallaxes)[indices_blazhko_GAIA] #contains all GAIA parallaxes for star for which the Blazhko type is known
    
    # use the stars for which a parallax has been found
    common_cols = df.columns.intersection(create_GAIA_output["Starname"])
    # get indices of common columns in dataframe containing unique stars without NaNs
    common_col_indices = [df.columns.get_loc(i) for i in common_cols]
    # slice the dataframe based on the common columns
    df = df[common_cols]
    
    # overwrite stars, RRABC, BRR (only selecting useful columns)
    stars = common_cols.values
    BRR = df.loc[["B/RR","RRAB/RRC"]].values[0]
    RRABC = df.loc[["B/RR","RRAB/RRC"]].values[1]

    #---------------------------------------------------------------------------------------
    
    # Select a subset of all parameters in the original dataframe, containing the observed magnitudes
    ms_df = df.copy().loc[['Vmag','e_Vmag','Ksmag','e_Ksmag','W1mag','e_W1mag']]
    # Make a pandas dataframe containing the ufloats (float64) of the observed magnitudes in different passbands
    m_df = pd.DataFrame([unumpy.uarray(ms_df.iloc[range(0, len(ms_df.values), 2)].iloc[j].astype(np.float64).tolist()
    ,ms_df.iloc[range(1, len(ms_df.values), 2)].iloc[h].astype(np.float64).tolist()) for j,h in zip(range(len(ms_df.values)/2),range(len(ms_df.values)/2))]
    , index=['V','Ks','W1'],columns=stars)
    # Make sure df columns are sorted in correct way
    m_df = m_df.reindex(columns=stars)
    
    #---------------------------------------------------------------------------------------
    
    # set the directory of the txt-file containing your output from NASA/IPAC  INFRARED  SCIENCE  ARCHIVE Galactic Dust Reddening and Extinction 'query'
    csv_Directory_EBV = '/Users/jvb/Documents/Proposal2018/Input_creation/'
    # filename of the txt-file containing output from NASA/IPAC  INFRARED  SCIENCE  ARCHIVE Galactic Dust Reddening and Extinction 'query'
    csv_filename_EBV = 'Dustmap_output_CSV_table.txt'
 
    
    # read in the (IPAC-table format) txt-file containing interstellar reddening information from DUST TABLES
    EBV_data = ascii.read(csv_Directory_EBV+csv_filename_EBV)
    # convert to dataframe, name columns (stars)
    EBV_df = EBV_data.to_pandas().T
    EBV_df = EBV_df.iloc[:, common_col_indices_blazhko] # select only those stars that have blazhko information available
    EBV_df = EBV_df.iloc[:, common_col_indices] # select only those stars that have GAIA information available

    EBV_df.columns = stars
    # rename certain table/dataframe rows to readable rows for the main program, so that your dust map reddening information is ready to be processed!
    EBV_df = EBV_df.rename(index={"mean_E_B_V_SandF": "EBV_S_F", "stdev_E_B_V_SandF": "e_EBV_S_F","mean_E_B_V_SFD":"EBV_SFD","stdev_E_B_V_SFD":"e_EBV_SFD"},columns=str)


#############################USING CSV INPUT################################### 


#---------------------------------------------------------------------------------------

# concatenate the dataframes
if our_script:
    ext_df_input = pd.concat([EBV_df,script_ext_df,df.loc[['RRAB/RRC']]])
else:
    ext_df_input = pd.concat([EBV_df,df.loc[['RRAB/RRC']]])

#---------------------------------------------------------------------------------------

# Utility definitions that allow one to look for functions in other modules (retrieve a specific function from an external module) and print all the functions available in an external module

def Search_function_Module(first_specifier,module,second_specifier=False):
    # find the function in a module, based on a first specifier and a second optional specifier
    
    # get all functions from the target module
    all_functions = inspect.getmembers(module, inspect.isfunction)
    # list all the names and all the functions
    list_names_functions = [l[0] for l in all_functions]
    list_functions = [y[1] for y in all_functions]
    # get indices of all functions containing first specifier in their name
    indices = [i for i, s in enumerate(list_names_functions) if first_specifier.lower() in s.lower()]
    # index the functions
    real_names_functions = [list_names_functions[j] for j in indices]
    real_functions = [list_functions[z] for z in indices]
    # check whether the second identifier is a string: if so index the 
    # indexed functionslist based on the second specifier
    if isinstance(second_specifier, basestring):
        indices_new = [i for i, s in enumerate(real_names_functions) if second_specifier.lower() in s.lower()]
        real_functions_new = [real_functions[p] for p in indices_new]
        return real_functions_new # return the wanted function from the external module
    else:
        return real_functions # return the wanted function from the external module

def print_functions_module(module):
    # get all functions from the target module
    all_functions = inspect.getmembers(module, inspect.isfunction)
    # print all the names and all the functions
    print("All the functions in the module " + str(module) + " are:")
    print(" ")
    for l in all_functions:
        print(l[0] + "              with arguments: " + str(inspect.getargspec(l[1])[0]) )
    print(" ")
    return

#---------------------------------------------------------------------------------------

# functions that allow one to retrieve the absolute magnitudes of all the stars in the sample provided, for all PML relations implemented (thus far).

def Calculate_Absolute_Magnitude_PML(PML_first_specifier,PML_second_specifier,dataframe):
    # search for specific PML relation in PML relations module
    real_PML = Search_function_Module(PML_first_specifier,PML,second_specifier=PML_second_specifier)
    # obtain names of arguments of specific PML relation
    arguments = inspect.getargspec(real_PML[0])[0]
    # define the arguments of specific PML relation
    if 'Period' in arguments:
        Period = df.loc['P'].values
        ePeriod = df.loc['e_P'].values
    if 'FeH' in arguments:
        FeH = df.loc['Fe'].values
        eFeH = df.loc['e_Fe'].values
    if 'RRab' in arguments:
        RRab_string = df.loc['RRAB/RRC'].values
        RRab = [bool('RRAB' in m) for m in list(RRab_string)]
        if 'symmetric' in arguments:
            symmetric = list(np.ones_like(df.iloc[0].values)*True)
    argument_values = [eval(e) for e in arguments]
    # convert strings from eval (e.g. strings containing the period values) to floats
    if len(argument_values) > 1:
        for j in range(len(argument_values)):
            if isinstance(argument_values[j][0], basestring):
                argument_values[j] = argument_values[j].astype(np.float)
    else:
        if isinstance(argument_values[0], basestring):
            argument_values = argument_values.astype(np.float)
    # return the numpy array of Absolute Magnitudes for the stellar sample for the specific PML relation
    list_mags = [real_PML[0](*[item[t] for item in list(argument_values)]) for t in range(len(argument_values[0]))]
    if re.match('(Sesar)',PML_first_specifier):
        dummy = [[uf,stars[r]] for r,uf in enumerate(list_mags) if np.isnan(uf.nominal_value)==False]
        names = [dummy[p][1] for p in range(len(dummy))]
        data = [dummy[p][0] for p in range(len(dummy))]
        return pd.DataFrame(data=data,index=names,columns=[PML_first_specifier + " " + PML_second_specifier]).T
    else:
        return pd.DataFrame(data=list_mags,index=stars,columns=[PML_first_specifier + " " + PML_second_specifier]).T

def get_all_absolute_magnitudes(df):
    specifier_list = ["Dambis","Dambis","Dambis","Klein","Klein","Muraveva","Muraveva","Neeley","Sesar"]
    specifier_list2 = ["V","Ks","W1","Mv","W1","LMC","plx","",""]
    magnitude_list = [Calculate_Absolute_Magnitude_PML(h,g,df) for h,g in zip(specifier_list,specifier_list2)]
    result_df = pd.concat([magnitude_list[0],magnitude_list[1]])
    for i in range(2,len(magnitude_list)):
        result_df = pd.concat([result_df,magnitude_list[i]])
    return result_df 

#---------------------------------------------------------------------------------------

# functions that allow one to compare the parallaxes obtained with each method (EXCEPT for parallaxes obtained with the Sesar method, since this relation is only valid for RRab RRLyr stars)

def Generate_pair_grid(Relations_indices):
    # List of keywords needed to obtain parallaxes for all relations
    Relations_list = ['Dambis V','Dambis Ks','Dambis W1','Klein Mv','Klein W1','Muraveva LMC','Muraveva plx','Neeley ','Sesar ']
    # calculate the parallaxes for all selected PML relations
    all_plxs = [Dplx.Distance_to_PLX(Dplx.Distance_modulus_distances(M_df,m_df,ext_df,rel,False)) for rel in [Relations_list[i] for i in Relations_indices]]
    # plot the corresponding pair grids for the different dereddening methods!
    BA.Pair_grid(all_plxs,df,uGAIAparallaxes)
    return

def plot_pair_grid(indices):
    Generate_pair_grid(indices)
    sys.exit()
    return

#---------------------------------------------------------------------------------------

# function that allows one to print the distances obtained with each method

def print_dist():   
    PML_relations = ['Dambis V','Dambis Ks','Dambis W1','Klein Mv','Klein W1','Muraveva LMC','Muraveva plx','Neeley ','Sesar ']
    Distances = [Dplx.Distance_modulus_distances(M_df,m_df,ext_df,j,False) for j in PML_relations]
    relations_dict =	{"Dambis V":"Dambis relation for the V-band","Dambis Ks":"Dambis relation for the Ks-band",
                      "Dambis W1":"Dambis relation for the W1-band","Klein Mv":"Klein relation for the V-band",
                      "Klein W1":"Klein relation for the W1-band","Muraveva LMC":"Muraveva relation for the Ks-band (based on LMC)",
                      "Muraveva plx":"Muraveva relation for the Ks-band (based on parallax)","Neeley ":"Neeley relation for the W1-band",
                      "Sesar ":"Sesar relation for the W1 band (only applicable to RRab, displaying Nan +/- Nan for RRc)"}
    for i,dist in enumerate(Distances):
        print("The Distances for the " + relations_dict[PML_relations[i]] + " are:")
        print(dist)
    sys.exit()
    return

# function that allows one to print the parallaxes obtained with each method

def print_plx():
    PML_relations = ['Dambis V','Dambis Ks','Dambis W1','Klein Mv','Klein W1','Muraveva LMC','Muraveva plx','Neeley ','Sesar ']
    Distances = [Dplx.Distance_modulus_distances(M_df,m_df,ext_df,j,False) for j in PML_relations]
    plxs = [Dplx.Distance_to_PLX(z,RRABC=RRABC,BRR=BRR,GAIA=uGAIAparallaxes) for z in Distances]
    relations_dict =	{"Dambis V":"Dambis relation for the V-band","Dambis Ks":"Dambis relation for the Ks-band",
                      "Dambis W1":"Dambis relation for the W1-band","Klein Mv":"Klein relation for the V-band",
                      "Klein W1":"Klein relation for the W1-band","Muraveva LMC":"Muraveva relation for the Ks-band (based on LMC)",
                      "Muraveva plx":"Muraveva relation for the Ks-band (based on parallax)","Neeley ":"Neeley relation for the W1-band",
                      "Sesar ":"Sesar relation for the W1 band (only applicable to RRab, displaying Nan +/- Nan for RRc)"}
    for i,plx in enumerate(plxs):
        print("The Parallaxes for the " + relations_dict[PML_relations[i]] + " are:")
        print(plx)
    sys.exit()
    return


#---------------------------------------------------------------------------------------
#                                   Main Script
#---------------------------------------------------------------------------------------

# generate dataframe containing all absolute magnitudes for the sample set of stars
M_df = get_all_absolute_magnitudes(df)
# Make sure df columns are sorted in correct way
M_df=M_df.reindex(columns=stars)

# generate dataframe containing different extinctions for the sample set of stars
ext_df = Dplx.get_extinctions(ext_df_input,R_error,our_script)
# Make sure df columns are sorted in correct way
ext_df=ext_df.reindex(columns=stars)

# print the distances obtained for all the stars when using our set of PML relations, if True
if distanceprint:
    print_dist()

# print the parallaxes obtained for the stars when using our set of PML relations, if True
if parallaxprint:
    print_plx()

# create the GAIA dataframe
GAIA_df = pd.DataFrame(uGAIAparallaxes,index=stars,columns=["GAIA"])
# print out any negative GAIA parallaxes and stop the script, if True
if print_neg_GAIA:
    print("------------------------------------------------------------------")
    print(GAIA_df[GAIA_df["GAIA"]<0])
    print("------------------------------------------------------------------")
    sys.exit()
# print out the number of RRAB/RRC and Blazhko/non-Blazhko variable stars   
if print_BRR_RRAB:
    print("------------------------------------------------------------------")
    print("# RRAB = " + str(len(df.T[df.T["RRAB/RRC"] == 'RRAB']["RRAB/RRC"])))
    print("# RRC = " + str(len(df.T[df.T["RRAB/RRC"] == 'RRC']["RRAB/RRC"])))
    print("# Blazhko = " + str(len(df.T[df.T["B/RR"] == 'Blazhko']["B/RR"])))
    print("# non-Blazhko = " + str(len(df.T[df.T["B/RR"] == 'RRLyr']["B/RR"])))
    print("------------------------------------------------------------------")
    sys.exit()

# MAIN 'LOOP'
if len(second_relation) == 0:
    Distances = Dplx.Distance_modulus_distances(M_df,m_df,ext_df,first_relation,False)
    df_plx = Dplx.Distance_to_PLX(Distances,RRABC=RRABC,BRR=BRR,GAIA=uGAIAparallaxes)
    if BLANDALTMAN:
        BA.Bland_Altman_main(df_plx,first_relation,second_relation,our_script,outfile_BA,percent=percent_BA , logplot=log_BA )
    if PASSINGBABLOK:
        PB.Passing_Bablok_Regression_Ref(df_plx,our_script,outfile_PB)
    
else:   
    Distances1,Distances2 = Dplx.Distance_modulus_distances(M_df,m_df,ext_df,first_relation,second_relation)
    df_plx = Dplx.Distance_to_PLX(Distances1,RRABC=RRABC,BRR=BRR,second_relation=True,df_distances2=Distances2)
    if BLANDALTMAN:
        BA.Bland_Altman_main(pd.concat([df_plx,GAIA_df.T]),first_relation,second_relation,our_script,outfile_BA,percent=percent_BA , logplot=log_BA)
    if PASSINGBABLOK:
        PB.Passing_Bablok_Regression_Ref(df_plx,our_script,outfile_PB)
    
plt.show() # show all plots
