## coding: utf-8
#!/usr/bin/env python2/3
#
# File: dereddening.py
# Author: Sven Nys <sven.nys@student.kuleuven.be>

# Description: Dereddening script used for the (manual) dereddening of RR Lyrae stars, depending on photometry scrutinously selected from literature.

# References:
# Castelli & Kurucz (2004): Castelli and Kurucz, New Grids of ATLAS9 Model Atmospheres, 2004, ArXiv e-prints, https://ui.adsabs.harvard.edu/#abs/2004astro.ph..5087C
# Cardelli et al. 1989: J. A. Cardelli et al. The relationship between infrared, optical, and ultraviolet extinction. Astrophysical Journal, Part 1, 345, 1989.

#   Copyright 2018 Sven Nys

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np

plt.close()

#Settings
#########################
# E(B-V) settings
N_ebv=1000 # nr. of intervals
ebv_min=0.0001 # minimal E(B-V) value
ebv_max=1.0 # maximal E(B-V) value
N_mc = 250 # Monte Carlo iteration

# decide whether or not you want to create a csv table that might be retro-inspected
table_write = False

# decide whether or not you want to create a raw data plot
raw_data = False

# Files need to be named according to these presets
StarNames = ['SVEri', 'XZDra', 'SWAnd', 'RUPsc', 'XAri', 'BD184995', 'RZCep', 'V1057Cas']
index = 1 # choose element of above list: calculate reddening for this object


#Read in data and model
##########################
t = Table.read( StarNames[index]+'.vot', format='votable') 

# (t['_tabname'] != '') &

# Select data you don't want from vot-file (removing outliers): assuming you have downloaded photometric data from ViZier database.

# SV Eri
if index==0: 
	t_reduced =t[(t['_tabname'] != 'I/239/tyc_main') & (t['_tabname'] != 'J/ApJS/203/32/table4') & (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'II/336/apass9') & (t['_tabname'] != 'II/328/allwise') & (t['_tabname'] != 'II/311/wise') & (t['_tabname'] != 'J/MNRAS/463/4210/ucac4rpm') & (t['_tabname'] != 'J/AJ/151/59/table2') & (t['_tabname'] != 'I/331/apop') & (t['_tabname'] != 'I/327/cmc15')]
# XZ Dra
if index==1: 
	t_reduced =t[ (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'II/271A/patch2')]
# SW And
if index==2: 
	t_reduced =t[(t['_tabname'] != 'J/MNRAS/441/715/table1') & (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'II/271A/patch2') &  (t['_tabname'] != 'II/336/apass9') & (t['_tabname'] != 'I/239/tyc_main') & (t['_tabname'] != 'II/311/wise') & (t['_tabname'] != 'I/331/apop') & (t['_tabname'] != 'II/328/allwise') & (t['_tabname'] != 'J/AJ/151/59/table2') & (t['_tabname'] != 'J/MNRAS/463/4210/ucac4rpm') & (t['_tabname'] != 'I/340/ucac5') & (t['_tabname'] != 'I/327/cmc15')]
# RU Psc
if index==3: 
	t_reduced =t[ (t['_tabname'] != 'II/311/wise') & (t['_tabname'] != 'J/AJ/151/59/table2') & (t['_tabname'] != 'II/271A/patch2') & (t['_tabname'] != 'J/MNRAS/441/715/table1') & (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'J/MNRAS/471/770/table2') & (t['_tabname'] != 'I/327/cmc15') & (t['_tabname'] != 'J/MNRAS/396/553/table') & (t['_tabname'] != 'J/MNRAS/435/3206/table2') & (t['_tabname'] != 'II/246/out')]
# X Ari
if index==4: 
	t_reduced =t[(t['_tabname'] != 'II/349/ps1' ) & (t['_tabname'] != 'II/311/wise' ) & (t['_tabname'] != 'II/328/allwise' ) & (t['_tabname'] != 'II/336/apass9')& (t['_tabname'] != 'J/MNRAS/463/4210/ucac4rpm')]
# BD+184995
if index==5: 
	t_reduced =t[(t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'I/275/ac2002') & (t['_tabname'] != 'II/336/apass9') & (t['_tabname'] != 'II/311/wise') & (t['_tabname'] != 'II/328/allwise') & (t['_tabname'] != 'II/271A/patch2') & (t['_tabname'] != 'I/342/f3') & (t['_tabname'] != 'I/239/tyc_main') & (t['_tabname'] != 'I/331/apop') & (t['_tabname'] != 'J/MNRAS/471/770/table1')]
# RZ Cep
if index==6: 
	t_reduced =t[ (t['_tabname'] != 'J/MNRAS/441/715/table1') & (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'J/MNRAS/396/553/table') & (t['_tabname'] != 'J/MNRAS/435/3206/table2')]
# V1057 Cas
if index==7:
    t_reduced =t
# No reduction of amount of data needed, all datapoints seem to be valid (every datapoint is within 0.5 arcsec and no large errorbars)

# select only (non-negative) datapoints/fluxes that have associated nonnegative errors/standard deviations
t_reduced = t_reduced[(t_reduced['sed_flux'] > 0.0 ) & (t_reduced['sed_eflux'] >0.0)]

# write the votable to readable format (csv), if True
if table_write:
    t.write(StarNames[index]+'.csv', format='ascii', delimiter=',', overwrite= True) # to write the .vot file to csv for reading

# read in the dereddening model (model atmospheres from Castelli & Kurucz 2004) data (supplied in repository)
model_data = ascii.read(StarNames[index]+'Model.csv', format='csv')

fig=plt.figure()

# plot the raw data in logscale, if True
if raw_data:
    ax = plt.gca()
    ax.scatter((3.0*10**9)/t_reduced['sed_freq'],np.log10(t_reduced['sed_freq']*t_reduced['sed_flux']), marker='+', s = 40, c='r', label='Raw Data') #Raw data plot
    ax.set_xscale('log')

#Scaling to the Johnson J filter of the model data
####################################################
scaling_model = np.interp(1250.8, model_data['wavelength'], model_data['flux']) # model has no datapoint (flux) at 1250.8 nm, so interpolation is necessary

# obtain data to be scaled, obtain the scaling factor and subsequently scale the model
scaling_data = t[(t['sed_filter'] =='Johnson:J')]
scale_factor = np.mean(scaling_data['sed_flux'])/scaling_model
scaled_model = scale_factor*model_data['flux']

# plot the scaled model in logscale
bx = plt.gca()
bx.plot(10.0*model_data['wavelength'], np.log10((3*10**17/model_data['wavelength'])*scaled_model), linestyle='dashed' , c='green', label='Scaled Stellar Atmosphere Model') #Scaled model plot
bx.set_xscale('log')


# R and C need to match in wavelength : interpolate lineary using the provided points for R and C
########################################################################################################
t_reduced_nodust = t_reduced[((3*10**9)/t_reduced['sed_freq'] < 30000)] # obtain wavelength regions not affected by dust flux

Lambda = (3*10**9)/(t_reduced_nodust['sed_freq']) # convert to wavelength in Angstrom, sed_freq is in GHz

# read in the interstellar dereddening law (Cardelli et al. 1989), and use it to interpolate the R-factor
Reddening = ascii.read('Reddening.csv', format = 'csv')
R_interp = np.interp(Lambda,Reddening['Wavelength'],Reddening['R'])

# create the E(B-V) array
E_BV = np.linspace(ebv_min,ebv_max,N_ebv) 		# iteration values of E(B-V)
C = np.interp(Lambda,10.0*model_data['wavelength'],scaled_model) # interpolate/calculate the model fluxes at the wavelengths of the photometric data


# Monte-Carlo
##########################
sigmadata = np.std(t_reduced_nodust['sed_flux']) # obtain the errors on the fluxes
E_BV_min = np.zeros(N_mc) # initialize an array containing the minimal E(B-V) values
ChiSquare_min =  np.zeros(len(E_BV_min)) # initialize an array containing the minimal Chi-squared values

for j in range(N_mc): # loop over all E(B-V) values
	if j%25 == 0:
		print(j)
	ChiSquare = np.zeros(len(E_BV))
	t_reduced_flux = t_reduced_nodust['sed_flux']
	t_reduced_nodust_new = []
	for k in range(len(t_reduced_flux)): # every value of the flux gets shifted by a random value (taken from a normal error distribution)
		t_reduced_nodust_new.append(t_reduced_flux[k] + np.random.normal(0,sigmadata)*sigmadata)

	for i in range(len(E_BV)):
		A_lambda = R_interp*E_BV[i]	# coefficient used to deredden
		F = 10**(A_lambda/2.5)*np.array(t_reduced_nodust_new) # Dereddened flux in Jy
        F_error = 10**(A_lambda/2.5)*np.array(t_reduced_nodust['sed_eflux']) # error on the Dereddened flux in Jy
            ChiSquare[i] = sum(((F-C)/F_error)**2)/(len(F)-1) # Chi-squared value
		

    ChiSquare_min[j]=np.argmin(ChiSquare) # obtain minimal Chi-squared value
    E_BV_min[j] = E_BV[np.argmin(ChiSquare)] # obtain minimal E(B-V) value
	
# calculate mean, minimum, and standard deviation of the minimal E(B-V) distribution (assuming a normal distribution)
mean_E_BV_min = np.mean(E_BV_min)
sigma_E_BV_min = np.std(E_BV_min)
min_E_BV_min = min(E_BV_min)

print(' ' )
print('Mean(E_BV_min)  ')
print('_______________________________' )
print(mean_E_BV_min )

print(' ' )
print('sigma(E_BV_min) ')
print('_______________________________' )
print(sigma_E_BV_min )

print( ' ' )
print('Min(E_BV_min)  ')
print('_______________________________' )
print(min_E_BV_min )

# calculate/interpolate R from wavelengths of observed data
Lambda_total = (3*10**9)/(t_reduced['sed_freq'])
R_interp = np.interp(Lambda_total,Reddening['Wavelength'],Reddening['R'])
# calculate A (with corresponding error) for found E(B-V) to interpolate to required frequencies
A_lambda = R_interp*mean_E_BV_min
A_lambda_err = R_interp*sigma_E_BV_min	
F_min = 10**(A_lambda/2.5)*t_reduced['sed_flux'] # calculate dereddened flux

# plot the dereddened fluxes (logscale)
cx = plt.gca()
cx.scatter(Lambda_total,np.log10((3*10**18/Lambda_total)*F_min), marker ='+' , s = 40, c='b', label='Dereddened Flux, E(B-V) = %.4f'%(mean_E_BV_min)) # Dereddened flux corresponding to mean E(B-V); E(B-V) rounded to 4 decimals places
cx.set_xscale('log')
axes = plt.gca() 
plt.legend()
plt.xlabel('Wavelength in $\AA$')
plt.ylabel('Log(f* F)  (F in Jy)')
plt.ylim(0,20)

#Calculate interpolated values of  A for selected wavelengths
######################################################################
def find_nearest_indices(array, value):
    array = np.asarray(array)   
    idx = np.argmin(np.abs(array - value))
    for i in range(len(array)-idx):
        if array[idx+i] !=  array[idx]:
            higher = idx + i
            break
    for j in range(idx):
        if array[idx-j-1] != array[idx]:
            lower = idx - j -1
            break
    if array[idx]-value>0:
        return lower,idx
    else:
        return idx,higher
def use_nearest_interp_error(wavelength,Lambdas,A_lambdas,A_lambdas_err):
    # piecewise linear interpolant ------> cfr. np.interp
    # y-y1 = (y2-y1)/(x2-x1) * (x-x1)
    idx1,idx2 = find_nearest_indices(Lambdas,wavelength)
    x1 = Lambdas[idx1]
    x2 = Lambdas[idx2]
    y1 = A_lambdas[idx1]
    e_y1 = A_lambdas_err[idx1]
    y2 = A_lambdas[idx2]
    e_y2 = A_lambdas_err[idx2]
    rico = (y2-y1)/(x2-x1)
    e_rico = (1./(x2-x1))*np.sqrt(np.power(e_y1,2)+np.power(e_y2,2))
    scaledrico = (wavelength-x1)*rico
    e_scaledrico = (wavelength-x1)*e_rico
    y = y1 + scaledrico
    e_y = np.sqrt(np.power(e_scaledrico,2)+np.power(e_y1,2))
    return y,e_y

# PHOTOMETRIC PASSBANDS OF INTEREST FOR PML RELATIONS STUDIED (RR LYRAE STARS)
# Visual Band V: 551 nm
# Wise Band W1: 3.4micron (WIKI) vs. 3.32 (DUST-tool)
# 2MASS Band Ks: 2.15micron (WIKI) vs. 2.16 (DUST-tool)

# obtain sorted wavelength array
Lambda_total_sorted = np.sort((3*10**9)/(t_reduced['sed_freq']))
R_interp_sorted = np.interp(Lambda_total_sorted,Reddening['Wavelength'],Reddening['R']) # calculate R
# calculate A for sorted wavelengths
A_lambda_sorted = R_interp_sorted*mean_E_BV_min
A_lambda_err = R_interp_sorted*sigma_E_BV_min

# 3 calculations below are to check whether definitions are right (they are)
#A_V = np.interp(5510, Lambda_total_sorted, A_lambda_sorted) # Lambda_total is in angstrom
#A_W = np.interp(33200, Lambda_total_sorted, A_lambda_sorted)
#A_K = np.interp(21600, Lambda_total_sorted, A_lambda_sorted)

# use definitions in order to obtain error on A for the different passband wavelengths
A_V_def,e_A_V_def = use_nearest_interp_error(5510,Lambda_total_sorted,A_lambda_sorted,A_lambda_err) 
A_W_def,e_A_W_def = use_nearest_interp_error(33200,Lambda_total_sorted,A_lambda_sorted,A_lambda_err)
A_K_def,e_A_K_def = use_nearest_interp_error(21600,Lambda_total_sorted,A_lambda_sorted,A_lambda_err)
print(' ' )
print('A_V')
print('__________________' )
print(A_V)
print(' ' )
print('A_V_def')
print('__________________' )
print(A_V_def)
print(e_A_V_def)
print(' ' )
print('A_W')
print('__________________' )
print(A_W)
print(' ' )
print('A_W_def')
print('__________________' )
print(A_W_def)
print(e_A_W_def)
print(' ' )
print('A_K')
print('__________________' )
print(A_K)
print('A_K_def')
print('__________________' )
print(A_K_def)
print(e_A_K_def)

if Chi_Squared_Check:
    #Plots to check wether chi² actually looks like chi²
    fig2=plt.figure()
    # plot all the Chi-squared values in function of E(B-V)
    Chiplot=plt.gca()
    Chiplot.scatter(E_BV, ChiSquare)
    axes2 = plt.gca()
    plt.xlabel('E(B-V)')
    plt.ylabel('$\chi ^2$')
    # plot the minimal Chi-squared values in function of minimal E(B-V)
    fig3=plt.figure()
    ChiMinplot=plt.gca()
    ChiMinplot.scatter(E_BV_min, ChiSquare_min)


# Create datatypes to write data to csv file (readable in main script)
Names = ['Mean(E(b-V))','sigma(E(b-V))','min(E(b-V))','A_V','eA_V','A_W','eA_W','A_K','eA_K']
Data = [mean_E_BV_min, sigma_E_BV_min, min_E_BV_min, A_V_def, e_A_V_def, A_W_def, e_A_W_def, A_K_def, e_A_K_def,]

# Create csv file (readable in main script)
import csv
with open(StarNames[index]+'_output.csv', 'w') as f:
	  writer = csv.writer(f, delimiter='\t')
	  writer.writerows(zip(Names,Data))
	  
plt.show()
