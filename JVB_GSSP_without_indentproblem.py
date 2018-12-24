# Written for Python 2.7/3.7 ----> HAVE TO CHECK

# Script that allows you to do Fundamental Parameter estimation by fitting these parameters (using the polynomial you choose) from the Chi-squared distribution obtained as output from GSSP (i.e. the Chi-squared table.dat)

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

################################################################################################
#               Written by Jordan Van Beeck - Based on Code by Nicolas Moens                   #
################################################################################################

################################################################################################
#                                      Dictionaries                                            #
################################################################################################

# Dictionaries used for calculation of Fundamental Parameter estimates (from GSSP Chi_2 table)

Parameters_GSSP_column_dictionary = {"Metallicity":0, "Teff":1, "log_g":2, "Microturbulent_velocity":3, "v_sin_i":5, "Chi2":8, "Chi2_err":9} # numbers correspond to column number in Chi2table.dat --> adjust to your liking

Units_Parameters_dictionary = {"Metallicity":"dex", "Teff":"K", "log_g":"dex", "Microturbulent_velocity":"km/s", "v_sin_i":"km/s"} # units of respective fundamental parameters determined in GSSP

################################################################################################
#                                       Definitions                                            #
################################################################################################

# Obtain the values of a fundamental parameter based on the chi2-table.dat file outputted by GSSP:

# REQUIRED ARGUMENTS
# parameters = which fundamental parameters you want to obtain/fit (list of strings)
# polydegree = which degree of polynomial should be used in the chi2table fit (integer) ----> e.g. when using only three values: don't chose a third degree polynomial...
# fitplot = determine whether or not you want plots of the fitted polynomial to the local Chi2-distribution (boolean)
# weights = determine whether or not you want a weighted polynomial fit, if None: non-weighted, if True: use inverse power (defined by weight power, standard value = 1) of chi-squared as weight
# averaging = determine whether or not you want to do a weighted polynomial fit taking into account Chi2-averaged values of the fundamental parameters ---> Don't use at this moment!!!!!

# OPTIONAL ARGUMENTS
# parmin = select whether or not the point of the GSSP parameter grid with lowest chi-squared value is taken as the minimum for the 1 sigma error estimation (if True); or use the fit minimum for the 1 sigma error calculation (if False; DEFAULT)
# within bounds = select whether or not you would like the polynomial fit to take place within the bounds of the GSSP parameter grid (i.e. the last and first grid points are the edges of the fit, if True); or expand the grid by 3 grid spacing factor on both sides (if False; DEFAULT)
# weight_power = integer that selects the power of your weights (i.e. -1 will take weightfactors^{-1} as weights), DEFAULT VALUE = -1 (Will only get used if the weights option is True)
# param_in_bound = if True, look for minimum only within the parameter range supplied to GSSP (i.e. between min and max of parameter range), default value = False
# likelihood = if True, use the Chi-Squared likelihood (exp^(-Chi^2)) value as weight for the fit points in the polynomial fit, default value = True

def get_plot_fundamental_parameters(parameters,polydegree,fitplot,weights,averaging,parmin=False,within_bounds=False,weight_power = -1.,param_in_bound=False,likelihood=True):
    column_numbers = []
    for p in np.array(parameters):
        column_numbers.append(Parameters_GSSP_column_dictionary[p])    
    # try to make numpy array containing all results
    allresults = np.zeros((len(star_name_array), len(column_numbers)))
    allresultsmin = np.zeros((len(star_name_array), len(column_numbers)))
    allresultsplus = np.zeros((len(star_name_array), len(column_numbers)))
    # convert it into a numpy array
    column_numbers = np.array(column_numbers)
    # loop over all stars
    for star_number in range(len(star_name_array)):
        data = pd.read_csv(Directory_chi2 + star_name_array[star_number] + '/Chi2_table.dat', sep = '\s+', header=None) # read in the full chi2-table using pandas (whitespace-separated)
        results = []
        resultsmin = []
        resultsplus = []
        for i in range(len(column_numbers)): # select the different fundamental parameters
            # chi2 average the fundamental parameters or not
            if averaging:
                par,ch,ch_dev = extract_average(data.iloc[:,column_numbers[i]].values, data.iloc[:,Parameters_GSSP_column_dictionary["Chi2"]].values)
                if likelihood:
                    real_weights = np.exp(-np.array(ch_dev))
                else:
                    real_weights = (np.array(ch_dev))**(weight_power)
            else:
                par,ch = extract_smallest(data.iloc[:,column_numbers[i]].values, data.iloc[:,Parameters_GSSP_column_dictionary["Chi2"]].values) # obtain the different unique parameter values and corresponding smallest Chi2 values!
                # use weights for fit or not
                if weights:
                    if likelihood:
                        real_weights = np.exp(-np.array(ch))
                    else:    
                        real_weights = (np.array(ch))**(weight_power)
                else:
                    real_weights = None
            if isinstance(polydegree, int):
                abscis,fit = make_fit(par,ch,polydegree,real_weights,within_bounds) # fit the polynomial of chosen degree to the (minimum) chi2 distribution of the fundamental parameter
            elif len(polydegree) == len(parameters):
                abscis,fit = make_fit(par,ch,polydegree[i],real_weights,within_bounds)
            else: 
                print("The length of your parameters list and polydegree list do not match, please revise this!")
                return 0,0,0,parameters
            c_err = [data.iloc[:,Parameters_GSSP_column_dictionary["Chi2_err"]].values[0]]*len(abscis) # define the one-sigma level from GSSP
            if parmin:
                negmin,posmin = sigma_error(abscis,fit,c_err,par[0],par,param_in_bound) # obtain the one-sigma errors for your estimates from GSSP 1 sigma estimate of chi2 (taking into account correlations);use par[0] as center because it has lowest chi2
            else:
                if param_in_bound:
                    search_fit_range = np.array(fit)[np.logical_and(abscis>=np.min(par),abscis<=np.max(par))] # only seek for minimum within parameter range
                    negmin,posmin = sigma_error(abscis,fit,c_err,abscis[fit.index(min(search_fit_range))],par,param_in_bound) # obtain the one-sigma errors for your estimates from GSSP 1 sigma estimate of chi2 (taking into account correlations);use minimum of fit as center
                else:   
                    negmin,posmin = sigma_error(abscis,fit,c_err,abscis[fit.index(min(fit))],par,param_in_bound) # obtain the one-sigma errors for your estimates from GSSP 1 sigma estimate of chi2 (taking into account correlations);use minimum of fit as center
            if param_in_bound:
                search_fit_range = np.array(fit)[np.logical_and(abscis>=np.min(par),abscis<=np.max(par))] # only seek for minimum within parameter range
                results.append(abscis[fit.index(min(search_fit_range))]) # obtain the minimal point of the polynomial fit, which is the estimated center/minimum chi2 parameter value from the fit
            else:
                results.append(abscis[fit.index(min(fit))]) # obtain the minimal point of the polynomial fit, which is the estimated center/minimum chi2 parameter value from the fit
            # append the 1 sigma errors for your estimates to the corresponding lists
            resultsmin.append(negmin)
            resultsplus.append(posmin)
            if fitplot: # make a plot of the fits for your star
                if isinstance(polydegree, int):
                    plot_fit_figure(polydegree,par,ch,abscis,fit,c_err,column_numbers[i],weights,weight_power,star_number,likelihood)
                else:
                    plot_fit_figure(polydegree[i],par,ch,abscis,fit,c_err,column_numbers[i],weights,weight_power,star_number,likelihood)
        # convert all stellar results to numpy arrays and write in corresponding allresults numpy arrays
        allresults[star_number] = np.array(results)
        allresultsmin[star_number] = np.array(resultsmin)
        allresultsplus[star_number] = np.array(resultsplus)
    return allresults,allresultsmin,allresultsplus,parameters

# plot the fit figures

def plot_fit_figure(polydegree,par,ch,abscis,fit,c_err,i,weights,weight_power,star_number,likelihood):
    plt.figure()
    if weights:
        if likelihood:
            plt.title("Polynomial fit (degree = " + str(polydegree) + ") of " + r'$\chi^2$' + " distribution of " + Parameters_GSSP_column_dictionary.keys()[Parameters_GSSP_column_dictionary.values().index(i)].replace("_", " ") + '\n (' + str(star_name_array[star_number]).replace("_", " ") +', weights: Likelihood)')
        else:
            plt.title("Polynomial fit (degree = " + str(polydegree) + ") of " + r'$\chi^2$' + " distribution of " + Parameters_GSSP_column_dictionary.keys()[Parameters_GSSP_column_dictionary.values().index(i)].replace("_", " ") + '\n (' + str(star_name_array[star_number]).replace("_", " ") +', weight power: '+ str(weight_power) +')')
    else:
        plt.title("Polynomial fit (degree = " + str(polydegree) + ") of " + r'$\chi^2$' + " distribution of " + Parameters_GSSP_column_dictionary.keys()[Parameters_GSSP_column_dictionary.values().index(i)].replace("_", " ") + '\n (' + str(star_name_array[star_number]).replace("_", " ") +')')
    plt.scatter(par,ch,label= r'$\chi^2$' + " Table",c='orange')
    plt.plot(abscis,fit,label= r'$\chi^2$' + " Fit",c='blue')
    plt.axhline(y=c_err[0], color='r', linestyle='-', label= r'$\chi^2$' + ' 1 ' + r'$\sigma$')
    plt.legend()
    plt.xlabel(Parameters_GSSP_column_dictionary.keys()[Parameters_GSSP_column_dictionary.values().index(i)].replace("_", " ") + " (" + Units_Parameters_dictionary[Parameters_GSSP_column_dictionary.keys()[Parameters_GSSP_column_dictionary.values().index(i)]].replace("_", " ") + ")")
    plt.ylabel(r'$\chi^2$')
    return


#Returns list of each parameter value in the grid with corresponding minimal chi2 --> in principle could be replaced using unique numpy array feature?
def extract_smallest(param,chi2):
    params = param.tolist()
    chi2s = chi2.tolist()
    param_smallest = []
    chi2_smallest = []
    while len(params) != 0:
        value = params[0]
        indices = get_indices(value,params)
        min_chi2 = 1000
        for j in indices:
            if chi2s[j] < min_chi2:
                min_chi2 = chi2s[j]
        remove_elements(indices,params)
        remove_elements(indices,chi2s)
        param_smallest.append(value)
        chi2_smallest.append(min_chi2)
    return param_smallest, chi2_smallest

# returns lists of each parameter value in the grid with corresponding average Chi2 and st_dev ---> problem with finding errors on estimates... ; probably neglecting correlations...
# Don't use at the moment!!!!!!
def extract_average(param,chi2):
    unique_values, indices = np.unique(param,return_inverse=True)
    average_ch = np.zeros_like(unique_values)
    stdev_ch = np.zeros_like(unique_values)
    for index, param in enumerate(unique_values):
        index_unique = np.where(indices==index)
        average_ch[index] = np.average(chi2[index_unique])
        stdev_ch[index] = np.std(chi2[index_unique])
    return unique_values, average_ch, stdev_ch

#Returns list of indices that have val as value
def get_indices(val,param):
    indices = [i for i in range(0,len(param)) if param[i]==val]
    return indices

#Removes all elements (of list) at the given indices
def remove_elements(indices,lis):
    for i in list(reversed(indices)):
        del lis[i]

#Make and plot polynomial fit from given x and y list --------> CAN change the fitting method but not recommended
def make_fit(x,y,degree,weights,within_bounds,method="poly"): # polynomial fit = default
    if within_bounds:
        # make linspace within bounds of the parameters (aka fit grid)
        abscis = np.linspace(min(x),max(x),1000)
    else:
        # define grid step of fit grid
        grid_step = abs(x[0]-x[1])
        # make linspace (aka fit grid)
        abscis = np.linspace(min(x)-3*grid_step,max(x)+3*grid_step,1000)
    if method == "poly":
        # polynomial fit of chosen degree, taking into account weights if wanted
        P = np.polynomial.polynomial.polyfit(x,y,degree,w=weights)
    elif method == "cheby":
        P = np.polynomial.chebyshev.chebfit(np.array(x),np.array(y),degree,w=weights)
    elif method == "legy":
        P = np.polynomial.legendre.legfit(np.array(x),np.array(y),degree,w=weights)
    elif method == "lagy":
        P = np.polynomial.laguerre.lagfit(np.array(x),np.array(y),degree,w=weights)
    elif method == "hermy":
        P = np.polynomial.hermite.hermfit(np.array(x),np.array(y),degree,w=weights)    

    # reverse the list in order to get list of highest number coefficients to lowest number coefficients (e.g. for 2nd order polynomial C_2 x^2 + C_1 x^1 + C_0 the coefficients in P are put into the following order: [C_2,C_1,C_0] instead of [C_0,C_1,C_2] which is the 'normal' output from the polynomial fit) 
    P = P[::-1]
    # define list of size abscis, with default value 0 (to be overwritten using list comprehension)
    fit = [0]*len(abscis)
    for i in range(len(P)):
        fit = [f+P[i]*l**(len(P)-1-i) for l,f in zip(abscis,fit)] # take the actual fit values
    return abscis, fit

#Calculate 1-sigma error (including correlation) on x variable
def sigma_error(xfit,yfit,yerr,centerx,par,param_in_bound):
    # generate residuals compared to 1 sigma value of chi2
    diff = [a-b for a,b in zip(yerr,yfit)]
    # look for zero crossings of residuals (i.e. where the fit intersects with the 1 sigma line)
    zeros = [xfit[i] for i in range(len(diff)-1) if diff[i]*diff[i+1]<0]
    # This gives the same result as:
    # xfit[np.argwhere(np.diff(np.sign(np.array(yerr) - np.array(yfit)))).flatten()]
    # if we enforce the minimum and side lobes to be inside the covered parameter range, remove zeros found not within bounds
    if param_in_bound:
        zeros = [i for i in zeros if (i<=np.max(par) and i>=np.min(par))]
    # generate the actual boundaries (i.e. '1 sigma error')
    negmin,posmin = boundaries(zeros,centerx)
    return negmin, posmin

#Returns smallest value larger and smaller than the center value (or 10000 if not found)
def boundaries(zeros,centerval):
    A = [z-centerval for z in zeros]
    # SET ARTIFICIAL BOUNDARIES FOR FUNDAMENTAL PARAMETER ESTIMATES ------> WILL NOT BE CHANGED IF GRID NOT WIDE ENOUGH e.g.
    posmin = 10000
    negmin = -10000
    # look for zero crossings 
    for i in range(len(A)):
        if A[i]>0 and A[i]<posmin:
            posmin = zeros[i]
        elif A[i]<0 and abs(A[i])<abs(negmin):
            negmin = zeros[i]
    return negmin, posmin

def printresults(allresults,allresultsmin,allresultsplus,parameters):
    # Print the fundamental parameter estimates and their corresponding 1 sigma errors
    if isinstance(allresults,int):
        if allresults == 0:
            print(" ")
            print("Something went wrong, please try again (see line above for error)")
        else: #added for safety
            print(" ")
            print(" Always Check Fit Plots in order to see whether you are experiencing a grid boundary problem!")
            print(" ")
            for star_number in range(len(star_name_array)):
                print("The Fundamental Parameter (" + str(parameters[0]) + ") of Star " + str(star_name_array[star_number]).replace("_", " ") + " is the following:")
                print("###########################################################################################################")
                print(str(parameters[0]).replace("_", " ") + ": " + "{:.4f}".format(allresults[star_number][0]) + ' (1 sigma +) ' + "{:.4f}".format(allresultsplus[star_number][0]) + ' (1 sigma -) ' + "{:.4f}".format(allresultsmin[star_number][0]) + " (delta +) " + "{:.4f}".format(allresultsplus[star_number][0] - allresults[star_number][0]) + " (delta -) " + "{:.4f}".format(allresults[star_number][0] - allresultsmin[star_number][0]))
                print("###########################################################################################################")
                print(" ")
                print(" ")
    elif len(parameters)==1: #only 1 listed parameter
        print(" ")
        print(" Always Check Fit Plots in order to see whether you are experiencing a grid boundary problem!")
        print(" ")
        for star_number in range(len(star_name_array)):
            print("The Fundamental Parameter of Star " + str(star_name_array[star_number]).replace("_", " ") + " are the following:")
            print("###########################################################################################################")
            for a in range(len(parameters)):
                print(str(parameters[a]).replace("_", " ") + ": " + "{:.4f}".format(allresults[star_number][a]) + ' (1 sigma +) ' + "{:.4f}".format(allresultsplus[star_number][a]) + ' (1 sigma -) ' + "{:.4f}".format(allresultsmin[star_number][a]) + " (delta +) " + "{:.4f}".format(allresultsplus[star_number][a] - allresults[star_number][a]) + " (delta -) " + "{:.4f}".format(allresults[star_number][a] - allresultsmin[star_number][a]))
            print("###########################################################################################################")
            print(" ")
            print(" ")          
    else: # go through all the listed parameters
        print(" ")
        print(" Always Check Fit Plots in order to see whether you are experiencing a grid boundary problem!")
        print(" ")
        for star_number in range(len(star_name_array)):
            print("The Fundamental Parameters of Star " + str(star_name_array[star_number]).replace("_", " ") + " are the following:")
            print("###########################################################################################################")
            for a in range(len(parameters)):
                print(str(parameters[a]).replace("_", " ") + ": " + "{:.4f}".format(allresults[star_number][a]) + ' (1 sigma +) ' + "{:.4f}".format(allresultsplus[star_number][a]) + ' (1 sigma -) ' + "{:.4f}".format(allresultsmin[star_number][a]) + " (delta +) " + "{:.4f}".format(allresultsplus[star_number][a] - allresults[star_number][a]) + " (delta -) " + "{:.4f}".format(allresults[star_number][a] - allresultsmin[star_number][a]))
            print("###########################################################################################################")
            print(" ")
            print(" ")            

################################################################################################
#                                       Main Script                                            #
################################################################################################

# Running the script on my IvS computer!
            
# Initial set-up: object names, folder containing chi2 tables

star_name_array = ['SW_AND','SW_AND_rough','V1057_Cas_rough'] # Change names of folder containing chi2 tables of stars

Directory_chi2 = '/home/jordanv/Observational_school_data_analysis/GSSP_single_v1.1/output_files/' # Name of folder in which chi2-tables are located ----> can either be relative or full path


# Running the script on my local computer!

star_name_array = ['XZ_Dra'] # Change names of folders containing chi2 tables of stars

Directory_chi2 = '/Users/jvb/Documents/Proposal2018/GSSP/' # Name of folder in which chi2-tables are located ----> can either be relative or full path

# plot (non-)weighted fits, and calculate the fundamental parameter estimates with their 1 sigma error estimates    


#       

# parameters,polydegree,fitplot,weights,averaging,parmin=False,within_bounds=False,weight_power = -1.,param_in_bound=False,likelihood=True
allresults,allresultsmin,allresultsplus,parameters = get_plot_fundamental_parameters(["Metallicity","Teff","log_g","Microturbulent_velocity","v_sin_i"],2,True,True,False,within_bounds=False,param_in_bound=False,weight_power = -1.)

# Print the fundamental parameter estimates and their corresponding 1 sigma errors    

printresults(allresults,allresultsmin,allresultsplus,parameters)