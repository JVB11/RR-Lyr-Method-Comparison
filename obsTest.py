from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np

plt.close()

#Settings
#########################
N_ebv=1000 #E(B-V) settings
ebv_min=0.0001
ebv_max=1.0
N_mc = 250 #MonteCarlo itteration

#Files need to be named according to these presets
StarNames = ['SVEri', 'XZDra', 'SWAnd', 'RUPsc', 'XAri', 'BD184995', 'RZCep', 'V1057Cas']
index = 0 #choose element of above list


#Read in data and model
##########################
t = Table.read( StarNames[index]+'.vot', format='votable') 

# (t['_tabname'] != '') &

#Select data you don't want from vot-file (removing outliers)

#SVEri 
#- good fit for high wavelenght, trouble fitting peak.
if index==0: 
	t_reduced =t[(t['_tabname'] != 'I/239/tyc_main') & (t['_tabname'] != 'J/ApJS/203/32/table4') & (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'II/336/apass9') & (t['_tabname'] != 'II/328/allwise') & (t['_tabname'] != 'II/311/wise') & (t['_tabname'] != 'J/MNRAS/463/4210/ucac4rpm') & (t['_tabname'] != 'J/AJ/151/59/table2') & (t['_tabname'] != 'I/331/apop') & (t['_tabname'] != 'I/327/cmc15')]
#XZDra - gives sensible fit
if index==1: 
	t_reduced =t[ (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'II/271A/patch2')]
#SWAnd - gives sensible fit
if index==2: 
	t_reduced =t[(t['_tabname'] != 'J/MNRAS/441/715/table1') & (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'II/271A/patch2') &  (t['_tabname'] != 'II/336/apass9') & (t['_tabname'] != 'I/239/tyc_main') & (t['_tabname'] != 'II/311/wise') & (t['_tabname'] != 'I/331/apop') & (t['_tabname'] != 'II/328/allwise') & (t['_tabname'] != 'J/AJ/151/59/table2') & (t['_tabname'] != 'J/MNRAS/463/4210/ucac4rpm') & (t['_tabname'] != 'I/340/ucac5') & (t['_tabname'] != 'I/327/cmc15')]
#RUPsc - It fits but the fit does seem a bit off and the E(B-V) is 10x smaller than expected, 0.05 instead of 0.5
if index==3: 
	t_reduced =t[ (t['_tabname'] != 'II/311/wise') & (t['_tabname'] != 'J/AJ/151/59/table2') & (t['_tabname'] != 'II/271A/patch2') & (t['_tabname'] != 'J/MNRAS/441/715/table1') & (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'J/MNRAS/471/770/table2') & (t['_tabname'] != 'I/327/cmc15') & (t['_tabname'] != 'J/MNRAS/396/553/table') & (t['_tabname'] != 'J/MNRAS/435/3206/table2') & (t['_tabname'] != 'II/246/out')]
#XAri - gives sensible fit
if index==4: 
	t_reduced =t[(t['_tabname'] != 'II/349/ps1' ) & (t['_tabname'] != 'II/311/wise' ) & (t['_tabname'] != 'II/328/allwise' ) & (t['_tabname'] != 'II/336/apass9')& (t['_tabname'] != 'J/MNRAS/463/4210/ucac4rpm')]
#Bd184995 - Not enough data to obtain model
if index==5: 
	t_reduced =t[(t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'I/275/ac2002') & (t['_tabname'] != 'II/336/apass9') & (t['_tabname'] != 'II/311/wise') & (t['_tabname'] != 'II/328/allwise') & (t['_tabname'] != 'II/271A/patch2') & (t['_tabname'] != 'I/342/f3') & (t['_tabname'] != 'I/239/tyc_main') & (t['_tabname'] != 'I/331/apop') & (t['_tabname'] != 'J/MNRAS/471/770/table1')]
#RZCep - It fits but the fit does seem a bit off and the E(B-V) is 10x smaller than expected, 0.09 instead of 0.9
if index==6: 
	t_reduced =t[ (t['_tabname'] != 'J/MNRAS/441/715/table1') & (t['_tabname'] != 'II/349/ps1') & (t['_tabname'] != 'J/MNRAS/396/553/table') & (t['_tabname'] != 'J/MNRAS/435/3206/table2')]
#V1057Cas - Not enough data to obtain model
if index==7:
    t_reduced =t
#No reduction needed, all data is good (every datapoint is within 0.5arcsec and no large errorbars)


t_reduced = t_reduced[(t_reduced['sed_flux'] > 0.0 ) & (t_reduced['sed_eflux'] >0.)]



#t.write(StarNames[index]+'.csv', format='ascii', delimiter=',', overwrite= True) #to write the .vot file to csv for reading
model_data = ascii.read(StarNames[index]+'Model.csv', format='csv')

fig=plt.figure()
#ax = plt.gca()
#ax.scatter((3*10**9)/t_reduced['sed_freq'],np.log10(t_reduced['sed_freq']*t_reduced['sed_flux']), marker='+', s = 40, c='r', label='Raw Data') #Raw data plot
#ax.set_xscale('log')

#Scaling to the filter Johnson J of the model data 
####################################################
scaling_model = np.interp(1250.8, model_data['wavelength'], model_data['flux'])#model data has no data at 1250.8 nm, so interpolating is necessary 

scaling_data = t[(t['sed_filter'] =='Johnson:J')]
scale_factor = np.mean(scaling_data['sed_flux'])/scaling_model
scaled_model = scale_factor*model_data['flux']


bx = plt.gca()
bx.plot(10*model_data['wavelength'], np.log10((3*10**17/model_data['wavelength'])*scaled_model), linestyle='dashed' , c='green', label='Scaled Stellar Atmosphere Model') #Scaled model plot
bx.set_xscale('log')


# R and C need to match in wavelength : interpolate lineary using the provided points for R and C
########################################################################################################
t_reduced_nodust = t_reduced[((3*10**9)/t_reduced['sed_freq'] < 30000)]

Lambda = (3*10**9)/(t_reduced_nodust['sed_freq']) #in angstrom, sed_freq is in GHz
Reddening = ascii.read('Reddening.csv', format = 'csv')
R_interp = np.interp(Lambda,Reddening['Wavelength'],Reddening['R'])
E_BV = np.linspace(ebv_min,ebv_max,N_ebv) 		#itteration values of E(B-V) 
C = np.interp(Lambda,10*model_data['wavelength'],scaled_model) 


# Monte-Carlo
##########################
sigmadata = np.std(t_reduced_nodust['sed_flux'])
E_BV_min = np.zeros(N_mc)
ChiSquare_min =  np.zeros(len(E_BV_min))

for j in range(N_mc):
	if j%25 == 0:
		print(j)
	ChiSquare = np.zeros(len(E_BV))
	t_reduced_flux = t_reduced_nodust['sed_flux']
	t_reduced_nodust_new = []
	for k in range(len(t_reduced_flux)): #every value of the flux gets shifted by a random value
		t_reduced_nodust_new.append(t_reduced_flux[k] + np.random.normal(0,sigmadata)*sigmadata)

	for i in range(len(E_BV)):
		A_lambda = R_interp*E_BV[i]	#coefficient to deredden
		F = 10**(A_lambda/2.5)*np.array(t_reduced_nodust_new) #Dereddened flux in Jy
		F_error = 10**(A_lambda/2.5)*np.array(t_reduced_nodust['sed_eflux'])
		ChiSquare[i] = sum(((F-C)/F_error)**2)/(len(F)-1)
		

	ChiSquare_min[j]=np.argmin(ChiSquare)
	E_BV_min[j] = E_BV[np.argmin(ChiSquare)]
	

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

Lambda_total = (3*10**9)/(t_reduced['sed_freq'])
R_interp = np.interp(Lambda_total,Reddening['Wavelength'],Reddening['R'])
A_lambda = R_interp*mean_E_BV_min #calculate A for found E(B-V) to interpolate to required frequencies
A_lambda_err = R_interp*sigma_E_BV_min	
F_min = 10**(A_lambda/2.5)*t_reduced['sed_flux']

cx = plt.gca()
cx.scatter(Lambda_total,np.log10((3*10**18/Lambda_total)*F_min), marker ='+' , s = 40, c='b', label='Dereddened Flux, E(B-V) = %.4f'%(mean_E_BV_min)) # Dereddened flux corresponding to mean E(B-V); E rounded to 4 decimals places
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
	 
#Visual Band V: 551 nm
#Wise Band W1: 3.4micron (WIKI) 3.32(DUST-tool)
#2MASS Band Ks: 2.15micron (WIKI) 2.16(DUST-tool)

Lambda_total_sorted = np.sort((3*10**9)/(t_reduced['sed_freq']))
R_interp_sorted = np.interp(Lambda_total_sorted,Reddening['Wavelength'],Reddening['R'])
A_lambda_sorted = R_interp_sorted*mean_E_BV_min
A_lambda_err = R_interp_sorted*sigma_E_BV_min
#3calcs below are to compare wether defs are right (they are)
A_V = np.interp(5510, Lambda_total_sorted, A_lambda_sorted) # Lambda_total is in angstrom
A_W = np.interp(33200, Lambda_total_sorted, A_lambda_sorted)
A_K = np.interp(21600, Lambda_total_sorted, A_lambda_sorted)

#definitions in order to obtain error
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

#Plots to check wether chi² actually looks like chi²
#fig2=plt.figure()
#
#Chiplot=plt.gca()
#Chiplot.scatter(E_BV, ChiSquare)
#axes2 = plt.gca() 
#plt.xlabel('E(B-V)')
#plt.ylabel('$\chi ^2$')
#
#fig3=plt.figure()
#ChiMinplot=plt.gca()
#ChiMinplot.scatter(E_BV_min, ChiSquare_min)

plt.show()

#Create datatypes to write data away
Names = ['Mean(E(b-V))','sigma(E(b-V))','min(E(b-V))','A_V','eA_V','A_W','eA_W','A_K','eA_K']
Data = [mean_E_BV_min, sigma_E_BV_min, min_E_BV_min, A_V_def, e_A_V_def, A_W_def, e_A_W_def, A_K_def, e_A_K_def,]

import csv
with open(StarNames[index]+'_output.csv', 'w') as f:
	  writer = csv.writer(f, delimiter='\t')
	  writer.writerows(zip(Names,Data))
