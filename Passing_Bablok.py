## coding: utf-8 
#!/usr/bin/env python2/3
#
# File: Passing_Bablok.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Module that implements the Passing Bablok regression 
# for the statistical comparison of the agreement of two methods.

# For more information on the method, see:
# Passing, H. & Bablok, W. (2009). A New Biometrical Procedure for Testing the Equality of Measurements 
# from Two Different Analytical Methods. Application of linear regression procedures for method comparison studies 
# in Clinical Chemistry, Part I. Clinical Chemistry and Laboratory Medicine, 21(11), pp. 709-720. 
# doi:10.1515/cclm.1983.21.11.70

# For more information on the bootstrap method, see:
# Carpenter, James and Bithell, John (2000). Bootstrap confidence intervals: 
# when, which, what? A practical guide for medical statisticians.
# in Statistics in Medicine, 19(9), pp. 1141-1164.
# doi:10.1002/(SICI)1097-0258(20000515)19:9<1141::AID-SIM479>3.0.CO;2-F

#   Copyright 2018 Jordan Van Beeck

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import math
import sys
import matplotlib.patches as mpatches

# add the uncertainties package!
from uncertainties import unumpy

sns.set_context("talk")

# import resampling tool for bootstrapping of confidence intervals --> sci-kit-learn
from sklearn.utils import resample

# Make a color dictionary, to be used during plotting
color_dict = { 'Blazhko':'orange', 'RRLyr':'blue'}
patchList = []
for key in color_dict:
        data_key = mpatches.Patch(color=color_dict[key], label=key)
        patchList.append(data_key)

def convert_uncert_df_to_nom_err(df):
    nom_list = []
    err_list = []
    for i in range(len(df)):
        values = df.iloc[i].values
        nom_list.append(unumpy.nominal_values(values))
        err_list.append(unumpy.std_devs(values))
    # generate dataframes containing the nominal values (nom) and the standard deviations (err)
    nom = pd.DataFrame(nom_list,index=list(df.index),columns=list(df))
    err = pd.DataFrame(err_list,index=list(df.index),columns=list(df))
    return nom,err

def Passing_Bablok_Regression_Ref(df,our_script,outfile):
    methods = list(df.index)
    if any("Sesar" in s for s in methods): # check whether Sesar script is used: if so, only look at RRab variables
        df = df.T[df.T["RRAB/RRC"]=="RRAB"].T
    df_copy = df.copy()
    df_copy = df_copy.drop(['RRAB/RRC','Blazhko/RRLyr'])    
    # obtain parallax values and corresponding errors
    nom_plx,err_plx = convert_uncert_df_to_nom_err(df_copy)
    if our_script:
        if len(nom_plx) > 4:
            methodlist1 = list(df_copy.index)[0:3] # first method
            methodlist2 = list(df_copy.index)[3:6] # second method to be compared against
            for i in range(len(methodlist1)):
                methodstring1 = methodlist1[i]
                methodstring2 = methodlist2[i]
                Compare = nom_plx.loc[methodstring1].values # parallaxes obtained by first method
                eCompare = err_plx.loc[methodstring1].values # errors
                Ref = nom_plx.loc[methodstring2].values # parallaxes obtained by second method
                # gathering non-extremal sample (see Tukey_Bland_Altman_Krouwer.py)
                differences = Ref - Compare
                non_extremal_indices = np.argwhere((differences < 1.) & (differences > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                non_extremal_Ref = Ref[non_extremal_indices]
                non_extremal_Compare = Compare[non_extremal_indices]
                non_extremal_eCompare = eCompare[non_extremal_indices] 
                # obtain the slopes
                slopes,sortedindices,sortedslopes,N = Slopes(Compare,Ref)
                non_extremal_slopes,non_extremal_sortedindices,non_extremal_sortedslopes,non_extremal_N = Slopes(non_extremal_Compare,non_extremal_Ref)
                # obtain the biased estimator of beta: 'b'
                b,Conf_bound_lower_beta,Conf_bound_higher_beta = estimate_beta(sortedslopes,Compare,N)
                non_extremal_b,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta = estimate_beta(non_extremal_sortedslopes,non_extremal_Compare,non_extremal_N)
                # obtain the estimate of alpha: 'a'
                a,a_conf_low,a_conf_high = estimate_alpha(Compare,Ref,Conf_bound_lower_beta,Conf_bound_higher_beta,b)
                non_extremal_a,non_extremal_a_conf_low,non_extremal_a_conf_high = estimate_alpha(non_extremal_Compare,non_extremal_Ref,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta,non_extremal_b)
                # Linearity check matrices: score and distance matrix, sorted; also obtain the fitted values
                sorted_distancematrix,sorted_scorematrix,fittedvalues,higher,lower,plx_indices = linearity_check_matrices(a,b,Compare,Ref)
                non_extremal_sorted_distancematrix,non_extremal_sorted_scorematrix,non_extremal_fittedvalues,non_extremal_higher,non_extremal_lower,non_extremal_plx_indices = linearity_check_matrices(non_extremal_a,non_extremal_b,non_extremal_Compare,non_extremal_Ref)
                # Sort colours, in order to make RRLyr/Blazhko differentiation
                sorted_colours = df.loc["Blazhko/RRLyr"].values[plx_indices]
                non_extremal_sorted_colours = df.T.iloc[non_extremal_indices].T.loc["Blazhko/RRLyr"].values[non_extremal_plx_indices]
                # calculating needed inputs for the plots assessing linearity --> ranks of distance matrix and the cusum statistic
                x = range(1,len(sorted_distancematrix)+1) # ranks of the distance matrix
                y = np.cumsum(sorted_scorematrix) # cumulative sum of the scores, needed for the cusum statistic
                non_extremal_x = range(1,len(non_extremal_sorted_distancematrix)+1) # ranks of the distance matrix
                non_extremal_y = np.cumsum(non_extremal_sorted_scorematrix) # cumulative sum of the scores, needed for the cusum statistic

                # estimate regression bounds by error propagation --> CRUDE APPROXIMATION
                low_bounds,high_bounds = estimate_regression_confidence(a,a_conf_low,a_conf_high,b,Conf_bound_lower_beta,Conf_bound_higher_beta,Compare,eCompare,fittedvalues)
                non_extremal_low_bounds,non_extremal_high_bounds = estimate_regression_confidence(non_extremal_a,non_extremal_a_conf_low,non_extremal_a_conf_high,non_extremal_b,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta,non_extremal_Compare,non_extremal_eCompare,non_extremal_fittedvalues)
                # Bootstrapping the confidence interval of fit:
                bootstraps,B = semi_param_resampling(Ref,Compare,a,b)
                bias = calculate_bias(Ref,Compare,a,b,bootstraps,B)
                Q = calculate_Q(B,bias)
                interval_bootstrap, neginterval_bootstrap = estimate_interval_pos(Ref,Compare,a,b,Q,B,bootstraps,interpolatedQ) # set interpolatedQ above
                interval_bootstrap, neginterval_bootstrap = estimate_interval_pos(Ref,Compare,a,b,Q,B,bootstraps,interpolatedQ) # set interpolatedQ above
                non_extremal_bootstraps,non_extremal_B = semi_param_resampling(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b)
                non_extremal_bias = calculate_bias(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b,non_extremal_bootstraps,non_extremal_B)
                non_extremal_Q = calculate_Q(non_extremal_B,non_extremal_bias)
                non_extremal_interval_bootstrap, non_extremal_neginterval_bootstrap = estimate_interval_pos(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b,non_extremal_Q,non_extremal_B,non_extremal_bootstraps,interpolatedQ) # set interpolatedQ above

                # plot the first Passing-Bablok plot: comparison with identity line:
                plot_comparison_identity(Compare,fittedvalues,low_bounds,high_bounds,methodstring1,interval_bootstrap,neginterval_bootstrap,methodstring2=methodstring2)
                plot_comparison_identity(non_extremal_Compare,non_extremal_fittedvalues,non_extremal_low_bounds,non_extremal_high_bounds,methodstring1,non_extremal_interval_bootstrap,non_extremal_neginterval_bootstrap,methodstring2=methodstring2,nonextremal=True)

                # plot the cusum statistic plots, assessing linearity:
                plot_cusum_statistic(True,x,y,higher,lower,sorted_colours,methodstring1,methodstring2=methodstring2)
                plot_cusum_statistic(False,x,y,higher,lower,sorted_colours,methodstring1,methodstring2=methodstring2)
                plot_cusum_statistic(True,non_extremal_x,non_extremal_y,non_extremal_higher,non_extremal_lower,non_extremal_sorted_colours,methodstring1,methodstring2=methodstring2,nonextremal=True)
                plot_cusum_statistic(False,non_extremal_x,non_extremal_y,non_extremal_higher,non_extremal_lower,non_extremal_sorted_colours,methodstring1,methodstring2=methodstring2,nonextremal=True)
                
                # Plot the regression residual plot, in terms of the rank
                plot_regression_residuals(x,Ref,fittedvalues,plx_indices,sorted_colours,methodstring1,methodstring2=methodstring2)
                plot_regression_residuals(non_extremal_x,non_extremal_Ref,non_extremal_fittedvalues,non_extremal_plx_indices,non_extremal_sorted_colours,methodstring1,methodstring2=methodstring2,nonextremal=True)
        
        else:
            Ref = nom_plx.loc["GAIA"].values # parallaxes obtained by GAIA method
            # eRef = err_plx.loc["GAIA"].values # errors
            # list the different dereddening methods
            methodlist = list(df_copy.index)[0:3]

            for i in range(len(methodlist)):
                methodstring = methodlist[i]
                Compare = nom_plx.loc[methodstring].values # parallaxes obtained by method to be compared
                eCompare = err_plx.loc[methodstring].values # errors
                # gathering non-extremal sample (see Tukey_Bland_Altman_Krouwer.py)
                differences = Ref - Compare
                non_extremal_indices = np.argwhere((differences < 1.) & (differences > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                non_extremal_Ref = Ref[non_extremal_indices]
                non_extremal_Compare = Compare[non_extremal_indices]
                non_extremal_eCompare = eCompare[non_extremal_indices] 
                # obtain the slopes
                slopes,sortedindices,sortedslopes,N = Slopes(Compare,Ref)
                non_extremal_slopes,non_extremal_sortedindices,non_extremal_sortedslopes,non_extremal_N = Slopes(non_extremal_Compare,non_extremal_Ref)
                # obtain the biased estimator of beta: 'b'
                b,Conf_bound_lower_beta,Conf_bound_higher_beta = estimate_beta(sortedslopes,Compare,N)
                non_extremal_b,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta = estimate_beta(non_extremal_sortedslopes,non_extremal_Compare,non_extremal_N)
                # obtain the estimate of alpha: 'a'
                a,a_conf_low,a_conf_high = estimate_alpha(Compare,Ref,Conf_bound_lower_beta,Conf_bound_higher_beta,b)
                non_extremal_a,non_extremal_a_conf_low,non_extremal_a_conf_high = estimate_alpha(non_extremal_Compare,non_extremal_Ref,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta,non_extremal_b)
                # Linearity check matrices: score and distance matrix, sorted; also obtain the fitted values
                sorted_distancematrix,sorted_scorematrix,fittedvalues,higher,lower,plx_indices = linearity_check_matrices(a,b,Compare,Ref)
                non_extremal_sorted_distancematrix,non_extremal_sorted_scorematrix,non_extremal_fittedvalues,non_extremal_higher,non_extremal_lower,non_extremal_plx_indices = linearity_check_matrices(non_extremal_a,non_extremal_b,non_extremal_Compare,non_extremal_Ref)
                # Sort colours, in order to make RRLyr/Blazhko differentiation
                sorted_colours = df.loc["Blazhko/RRLyr"].values[plx_indices]
                non_extremal_sorted_colours = df.T.iloc[non_extremal_indices].T.loc["Blazhko/RRLyr"].values[non_extremal_plx_indices]
                # calculating needed inputs for the plots assessing linearity --> ranks of distance matrix and the cusum statistic
                x = range(1,len(sorted_distancematrix)+1) # ranks of the distance matrix
                y = np.cumsum(sorted_scorematrix) # cumulative sum of the scores, needed for the cusum statistic
                non_extremal_x = range(1,len(non_extremal_sorted_distancematrix)+1) # ranks of the distance matrix
                non_extremal_y = np.cumsum(non_extremal_sorted_scorematrix) # cumulative sum of the scores, needed for the cusum statistic
        
                # estimate regression bounds by error propagation --> CRUDE APPROXIMATION
                low_bounds,high_bounds = estimate_regression_confidence(a,a_conf_low,a_conf_high,b,Conf_bound_lower_beta,Conf_bound_higher_beta,Compare,eCompare,fittedvalues)
                non_extremal_low_bounds,non_extremal_high_bounds = estimate_regression_confidence(non_extremal_a,non_extremal_a_conf_low,non_extremal_a_conf_high,non_extremal_b,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta,non_extremal_Compare,non_extremal_eCompare,non_extremal_fittedvalues)
                # Bootstrapping the confidence interval of fit:
                bootstraps,B = semi_param_resampling(Ref,Compare,a,b)
                bias = calculate_bias(Ref,Compare,a,b,bootstraps,B)
                Q = calculate_Q(B,bias)
                interval_bootstrap, neginterval_bootstrap = estimate_interval_pos(Ref,Compare,a,b,Q,B,bootstraps,interpolatedQ) # set interpolatedQ above
                non_extremal_bootstraps,non_extremal_B = semi_param_resampling(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b)
                non_extremal_bias = calculate_bias(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b,non_extremal_bootstraps,non_extremal_B)
                non_extremal_Q = calculate_Q(non_extremal_B,non_extremal_bias)
                non_extremal_interval_bootstrap, non_extremal_neginterval_bootstrap = estimate_interval_pos(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b,non_extremal_Q,non_extremal_B,non_extremal_bootstraps,interpolatedQ) # set interpolatedQ above
        
                # plot the first Passing-Bablok plot: comparison with identity line:
                plot_comparison_identity(Compare,fittedvalues,low_bounds,high_bounds,methodstring,interval_bootstrap,neginterval_bootstrap)
                plot_comparison_identity(non_extremal_Compare,non_extremal_fittedvalues,non_extremal_low_bounds,non_extremal_high_bounds,methodstring,non_extremal_interval_bootstrap,non_extremal_neginterval_bootstrap,nonextremal=True)
                # plot the cusum statistic plots, assessing linearity:
                plot_cusum_statistic(True,x,y,higher,lower,sorted_colours,methodstring)
                plot_cusum_statistic(False,x,y,higher,lower,sorted_colours,methodstring)
                plot_cusum_statistic(True,non_extremal_x,non_extremal_y,non_extremal_higher,non_extremal_lower,non_extremal_sorted_colours,methodstring,nonextremal=True)
                plot_cusum_statistic(False,non_extremal_x,non_extremal_y,non_extremal_higher,non_extremal_lower,non_extremal_sorted_colours,methodstring,nonextremal=True)
                
                # Plot the regression residual plot, in terms of the rank
                plot_regression_residuals(x,Ref,fittedvalues,plx_indices,sorted_colours,methodstring)
                plot_regression_residuals(non_extremal_x,non_extremal_Ref,non_extremal_fittedvalues,non_extremal_plx_indices,non_extremal_sorted_colours,methodstring,nonextremal=True)

    else:
        if len(nom_plx) > 3:
            methodlist1 = list(df_copy.index)[0:2] # first method
            methodlist2 = list(df_copy.index)[2:4] # second method to be compared against
            for i in range(len(methodlist1)):
                methodstring1 = methodlist1[i]
                methodstring2 = methodlist2[i]
                Compare = nom_plx.loc[methodstring1].values # parallaxes obtained by first method
                eCompare = err_plx.loc[methodstring1].values # errors
                Ref = nom_plx.loc[methodstring2].values # parallaxes obtained by second method
                # gathering non-extremal sample (see Tukey_Bland_Altman_Krouwer.py)
                differences = Ref - Compare
                non_extremal_indices = np.argwhere((differences < 1.) & (differences > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                non_extremal_Ref = Ref[non_extremal_indices]
                non_extremal_Compare = Compare[non_extremal_indices]
                non_extremal_eCompare = eCompare[non_extremal_indices] 
                # obtain the slopes
                slopes,sortedindices,sortedslopes,N = Slopes(Compare,Ref)
                non_extremal_slopes,non_extremal_sortedindices,non_extremal_sortedslopes,non_extremal_N = Slopes(non_extremal_Compare,non_extremal_Ref)
                # obtain the biased estimator of beta: 'b'
                b,Conf_bound_lower_beta,Conf_bound_higher_beta = estimate_beta(sortedslopes,Compare,N)
                non_extremal_b,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta = estimate_beta(non_extremal_sortedslopes,non_extremal_Compare,non_extremal_N)
                # obtain the estimate of alpha: 'a'
                a,a_conf_low,a_conf_high = estimate_alpha(Compare,Ref,Conf_bound_lower_beta,Conf_bound_higher_beta,b)
                non_extremal_a,non_extremal_a_conf_low,non_extremal_a_conf_high = estimate_alpha(non_extremal_Compare,non_extremal_Ref,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta,non_extremal_b)
                # Linearity check matrices: score and distance matrix, sorted; also obtain the fitted values
                sorted_distancematrix,sorted_scorematrix,fittedvalues,higher,lower,plx_indices = linearity_check_matrices(a,b,Compare,Ref)
                non_extremal_sorted_distancematrix,non_extremal_sorted_scorematrix,non_extremal_fittedvalues,non_extremal_higher,non_extremal_lower,non_extremal_plx_indices = linearity_check_matrices(non_extremal_a,non_extremal_b,non_extremal_Compare,non_extremal_Ref)
                # Sort colours, in order to make RRLyr/Blazhko differentiation
                sorted_colours = df.loc["Blazhko/RRLyr"].values[plx_indices]
                non_extremal_sorted_colours = df.T.iloc[non_extremal_indices].T.loc["Blazhko/RRLyr"].values[non_extremal_plx_indices]
                # calculating needed inputs for the plots assessing linearity --> ranks of distance matrix and the cusum statistic
                x = range(1,len(sorted_distancematrix)+1) # ranks of the distance matrix
                y = np.cumsum(sorted_scorematrix) # cumulative sum of the scores, needed for the cusum statistic
                non_extremal_x = range(1,len(non_extremal_sorted_distancematrix)+1) # ranks of the distance matrix
                non_extremal_y = np.cumsum(non_extremal_sorted_scorematrix) # cumulative sum of the scores, needed for the cusum statistic
                
                # estimate regression bounds by error propagation --> CRUDE APPROXIMATION
                low_bounds,high_bounds = estimate_regression_confidence(a,a_conf_low,a_conf_high,b,Conf_bound_lower_beta,Conf_bound_higher_beta,Compare,eCompare,fittedvalues)
                non_extremal_low_bounds,non_extremal_high_bounds = estimate_regression_confidence(non_extremal_a,non_extremal_a_conf_low,non_extremal_a_conf_high,non_extremal_b,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta,non_extremal_Compare,non_extremal_eCompare,non_extremal_fittedvalues)
                # Bootstrapping the confidence interval of fit:
                bootstraps,B = semi_param_resampling(Ref,Compare,a,b)
                bias = calculate_bias(Ref,Compare,a,b,bootstraps,B)
                Q = calculate_Q(B,bias)
                interval_bootstrap, neginterval_bootstrap = estimate_interval_pos(Ref,Compare,a,b,Q,B,bootstraps,interpolatedQ) # set interpolatedQ above
                non_extremal_bootstraps,non_extremal_B = semi_param_resampling(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b)
                non_extremal_bias = calculate_bias(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b,non_extremal_bootstraps,non_extremal_B)
                non_extremal_Q = calculate_Q(non_extremal_B,non_extremal_bias)
                non_extremal_interval_bootstrap, non_extremal_neginterval_bootstrap = estimate_interval_pos(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b,non_extremal_Q,non_extremal_B,non_extremal_bootstraps,interpolatedQ) # set interpolatedQ above
                
                # plot the first Passing-Bablok plot: comparison with identity line:
                plot_comparison_identity(Compare,fittedvalues,low_bounds,high_bounds,methodstring1,interval_bootstrap,neginterval_bootstrap,methodstring2=methodstring2)
                plot_comparison_identity(non_extremal_Compare,non_extremal_fittedvalues,non_extremal_low_bounds,non_extremal_high_bounds,methodstring1,non_extremal_interval_bootstrap,non_extremal_neginterval_bootstrap,methodstring2=methodstring2,nonextremal=True)

                # plot the cusum statistic plots, assessing linearity:
                plot_cusum_statistic(True,x,y,higher,lower,sorted_colours,methodstring1,methodstring2=methodstring2)
                plot_cusum_statistic(False,x,y,higher,lower,sorted_colours,methodstring1,methodstring2=methodstring2)
                plot_cusum_statistic(True,non_extremal_x,non_extremal_y,non_extremal_higher,non_extremal_lower,non_extremal_sorted_colours,methodstring1,methodstring2=methodstring2,nonextremal=True)
                plot_cusum_statistic(False,non_extremal_x,non_extremal_y,non_extremal_higher,non_extremal_lower,non_extremal_sorted_colours,methodstring1,methodstring2=methodstring2,nonextremal=True)
                
                # Plot the regression residual plot, in terms of the rank
                plot_regression_residuals(x,Ref,fittedvalues,plx_indices,sorted_colours,methodstring1,methodstring2=methodstring2)
                plot_regression_residuals(non_extremal_x,non_extremal_Ref,non_extremal_fittedvalues,non_extremal_plx_indices,non_extremal_sorted_colours,methodstring1,methodstring2=methodstring2,nonextremal=True)
    
        else:
            Ref = nom_plx.loc["GAIA"].values # parallaxes obtained by GAIA method            
            # eRef = err_plx.loc["GAIA"].values # errors
            # list the different dereddening methods
            methodlist = list(df_copy.index)[0:2]
            
            for i in range(len(methodlist)):
                methodstring = methodlist[i]
                Compare = nom_plx.loc[methodstring].values # parallaxes obtained by method to be compared
                eCompare = err_plx.loc[methodstring].values # errors                
                # gathering non-extremal sample (see Tukey_Bland_Altman_Krouwer.py)
                differences = Ref - Compare
                non_extremal_indices = np.argwhere((differences < 1.) & (differences > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                non_extremal_Ref = Ref[non_extremal_indices]
                non_extremal_Compare = Compare[non_extremal_indices]
                non_extremal_eCompare = eCompare[non_extremal_indices]                
                # obtain the slopes
                slopes,sortedindices,sortedslopes,N = Slopes(Compare,Ref)
                non_extremal_slopes,non_extremal_sortedindices,non_extremal_sortedslopes,non_extremal_N = Slopes(non_extremal_Compare,non_extremal_Ref)
                # obtain the biased estimator of beta: 'b'
                b,Conf_bound_lower_beta,Conf_bound_higher_beta = estimate_beta(sortedslopes,Compare,N)
                non_extremal_b,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta = estimate_beta(non_extremal_sortedslopes,non_extremal_Compare,non_extremal_N)
                # obtain the estimate of alpha: 'a'
                a,a_conf_low,a_conf_high = estimate_alpha(Compare,Ref,Conf_bound_lower_beta,Conf_bound_higher_beta,b)
                non_extremal_a,non_extremal_a_conf_low,non_extremal_a_conf_high = estimate_alpha(non_extremal_Compare,non_extremal_Ref,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta,non_extremal_b)
                # Linearity check matrices: score and distance matrix, sorted; also obtain the fitted values
                sorted_distancematrix,sorted_scorematrix,fittedvalues,higher,lower,plx_indices = linearity_check_matrices(a,b,Compare,Ref)
                non_extremal_sorted_distancematrix,non_extremal_sorted_scorematrix,non_extremal_fittedvalues,non_extremal_higher,non_extremal_lower,non_extremal_plx_indices = linearity_check_matrices(non_extremal_a,non_extremal_b,non_extremal_Compare,non_extremal_Ref)
                # Sort colours, in order to make RRLyr/Blazhko differentiation
                sorted_colours = df.loc["Blazhko/RRLyr"].values[plx_indices]
                non_extremal_sorted_colours = df.T.iloc[non_extremal_indices].T.loc["Blazhko/RRLyr"].values[non_extremal_plx_indices]
                # calculating needed inputs for the plots assessing linearity --> ranks of distance matrix and the cusum statistic
                x = range(1,len(sorted_distancematrix)+1) # ranks of the distance matrix
                y = np.cumsum(sorted_scorematrix) # cumulative sum of the scores, needed for the cusum statistic
                non_extremal_x = range(1,len(non_extremal_sorted_distancematrix)+1) # ranks of the distance matrix
                non_extremal_y = np.cumsum(non_extremal_sorted_scorematrix) # cumulative sum of the scores, needed for the cusum statistic
                
                # estimate regression bounds by error propagation --> CRUDE APPROXIMATION
                low_bounds,high_bounds = estimate_regression_confidence(a,a_conf_low,a_conf_high,b,Conf_bound_lower_beta,Conf_bound_higher_beta,Compare,eCompare,fittedvalues)
                non_extremal_low_bounds,non_extremal_high_bounds = estimate_regression_confidence(non_extremal_a,non_extremal_a_conf_low,non_extremal_a_conf_high,non_extremal_b,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta,non_extremal_Compare,non_extremal_eCompare,non_extremal_fittedvalues)
                # Bootstrapping the confidence interval of fit:
                bootstraps,B = semi_param_resampling(Ref,Compare,a,b)
                bias = calculate_bias(Ref,Compare,a,b,bootstraps,B)
                Q = calculate_Q(B,bias)
                interval_bootstrap, neginterval_bootstrap = estimate_interval_pos(Ref,Compare,a,b,Q,B,bootstraps,interpolatedQ) # set interpolatedQ above
                non_extremal_bootstraps,non_extremal_B = semi_param_resampling(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b)
                non_extremal_bias = calculate_bias(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b,non_extremal_bootstraps,non_extremal_B)
                non_extremal_Q = calculate_Q(non_extremal_B,non_extremal_bias)
                non_extremal_interval_bootstrap, non_extremal_neginterval_bootstrap = estimate_interval_pos(non_extremal_Ref,non_extremal_Compare,non_extremal_a,non_extremal_b,non_extremal_Q,non_extremal_B,non_extremal_bootstraps,interpolatedQ) # set interpolatedQ above
              
                # plot the first Passing-Bablok plot: comparison with identity line:
                plot_comparison_identity(Compare,fittedvalues,low_bounds,high_bounds,methodstring,interval_bootstrap,neginterval_bootstrap)
                plot_comparison_identity(non_extremal_Compare,non_extremal_fittedvalues,non_extremal_low_bounds,non_extremal_high_bounds,methodstring,non_extremal_interval_bootstrap,non_extremal_neginterval_bootstrap,nonextremal=True)
                # plot the cusum statistic plots, assessing linearity:
                plot_cusum_statistic(True,x,y,higher,lower,sorted_colours,methodstring)
                plot_cusum_statistic(False,x,y,higher,lower,sorted_colours,methodstring)
                plot_cusum_statistic(True,non_extremal_x,non_extremal_y,non_extremal_higher,non_extremal_lower,non_extremal_sorted_colours,methodstring,nonextremal=True)
                plot_cusum_statistic(False,non_extremal_x,non_extremal_y,non_extremal_higher,non_extremal_lower,non_extremal_sorted_colours,methodstring,nonextremal=True)
                
                # Plot the regression residual plot, in terms of the rank
                plot_regression_residuals(x,Ref,fittedvalues,plx_indices,sorted_colours,methodstring)
                plot_regression_residuals(non_extremal_x,non_extremal_Ref,non_extremal_fittedvalues,non_extremal_plx_indices,non_extremal_sorted_colours,methodstring,nonextremal=True)
                with open(outfile, 'w') as f:
                    print_write_output(a,a_conf_low,a_conf_high,b,Conf_bound_lower_beta,Conf_bound_higher_beta,f,methodstring)
                    print_write_output(non_extremal_a,non_extremal_a_conf_low,non_extremal_a_conf_high,non_extremal_b,non_extremal_Conf_bound_lower_beta,non_extremal_Conf_bound_higher_beta,f,methodstring,nonextremal=True)
    return 

def print_write_output(a,a_conf_low,a_conf_high,b,Conf_bound_lower_beta,Conf_bound_higher_beta,outfile,method1,method2=False,nonextremal=False):
    if isinstance(method2, basestring):
        whatisfitted =  method2 + " parallax" + " vs. " + method1 + " parallax"
    else:
        whatisfitted = 'GAIA parallax' + " vs. " + method1 + " parallax"
    if sys.version_info[0] < 3: # if python 2
        print >> outfile, "-----------------------------------------------------"
        print >> outfile, " "
        print >> outfile, " "
        if nonextremal:
            print >> outfile, "Estimated alpha and beta for Passing Bablok Regression (non-extremal;" + whatisfitted + "):" 
        else:
            print >> outfile, "Estimated alpha and beta for Passing Bablok Regression (" + whatisfitted + "):"
        print >> outfile, " "
        print >> outfile, " "
        print >> outfile, " estimated parameter: value (+) upper confidence bound (-) lower confidence bound "
        print >> outfile, " "
        print >> outfile, " "
        if nonextremal:
            print >> outfile, " alpha (non-extremal): " + str(a) + " (+) " + str(a_conf_high - a) + " (-) " + str(a - a_conf_low)
            print >> outfile, " beta (non-extremal): " + str(b) + " (+) " + str(Conf_bound_higher_beta - b) + " (-) " + str(b - Conf_bound_lower_beta)
        else:
            print >> outfile, " alpha: " + str(a) + " (+) " + str(a_conf_high - a) + " (-) " + str(a - a_conf_low)
            print >> outfile, " beta: " + str(b) + " (+) " + str(Conf_bound_higher_beta - b) + " (-) " + str(b - Conf_bound_lower_beta)
        print >> outfile, " "
        print >> outfile, " "
        print >> outfile, "-----------------------------------------------------"
    else: # python 3
        print("-----------------------------------------------------")
        print( " ")
        print(" ")
        if nonextremal:
            print("Estimated alpha and beta for Passing Bablok Regression (non-extremal; " + whatisfitted + "):" )
        else:
            print("Estimated alpha and beta for Passing Bablok Regression (" + whatisfitted + "):")
        print(" ")
        print(" ")
        print(" estimated parameter: value (+) upper confidence bound (-) lower confidence bound ")
        print(" ")
        print(" ")
        if nonextremal:
            print(" alpha (non-extremal): " + str(a) + " (+) " + str(a_conf_high - a) + " (-) " + str(a - a_conf_low))
            print(" beta (non-extremal): " + str(b) + " (+) " + str(Conf_bound_higher_beta - b) + " (-) " + str(b - Conf_bound_lower_beta))
        else:
            print(" alpha: " + str(a) + " (+) " + str(a_conf_high - a) + " (-) " + str(a - a_conf_low))
            print(" beta: " + str(b) + " (+) " + str(Conf_bound_higher_beta - b) + " (-) " + str(b - Conf_bound_lower_beta))
        print(" ")
        print(" ")
        print("-----------------------------------------------------" )
    return

def Slopes(X,Y): # Slope function needed for Theil's/Passing-Bablok procedure. 
    # --> calculating slopes using the unique pairs (x_i,y_j) where 1 <= i < j <= n
    slopes = []    
    for i in range(len(Y)):
        for j in range(i+1,len(Y)):
            # dismiss special cases which shouldn't occur frequently!
            if Y[i] != Y[j] and X[i] != X[j] and (Y[i]-Y[j])/(X[i]-X[j]) != -1: 
                slopes.append((Y[i]-Y[j])/(X[i]-X[j]))
    sortedslopes = np.sort(np.array(slopes)) # sort the slopes
    sortedindices = np.argsort(np.array(slopes)) # obtain the sorted indices
    N = len(slopes) # obtain the number of slopes calculated (i.e. non-special cases)
    return slopes,sortedindices,sortedslopes,N

def estimate_beta(sortedslopes,Compare,N):
    # obtain the parameters needed for the (biased) estimation of beta/obtaining the biased estimator 'b'
    K = len(sortedslopes[sortedslopes<-1])
    n = len(Compare)
    if N%2 != 0: # odd amount of slopes
        b = sortedslopes[K+ ((N+1)/2)]
    else: # even amount of slopes
        b = (1./2.)*(sortedslopes[K + N/2] + sortedslopes[K + 1 + N/2]) # best estimate = interpolation (in this case we just take the average)
      
    # calculate the corresponding (95%) confidence intervals of the estimator 'b'
    gamma = 0.05 # we're calculating the 95% confidence interval
    quantile_normal = norm.ppf(1.-(gamma/2.)) # get the (1-gamma/2) quantile of the standardnormal distribution
    C_gamma = quantile_normal * np.sqrt((n*(n-1.)*((2.*n) + 5.))/18.) 
    M_1 = (N - C_gamma)/2
    M_1 = int(round(M_1)) # make the result an integer
    M_2 = N - M_1 + 1
    Conf_bound_lower_beta = sortedslopes[K + M_1]
    if (K+M_2) <= len(sortedslopes)-1: 
        Conf_bound_higher_beta = sortedslopes[K + M_2]
    else: 
        Conf_bound_higher_beta = sortedslopes[len(sortedslopes)-1] # highest possible --> solves error that sometimes occurs
    return b,Conf_bound_lower_beta,Conf_bound_higher_beta

def estimate_alpha(Compare,Ref,Conf_bound_lower_beta,Conf_bound_higher_beta,b):
    # obtain the lists of medians needed in order to obtain the estimator 'a' of alpha, 
    # and the corresponding confidence intervals
    
    # estimate a for all stars
    a_estimators = [Ref[j]-(b*com) for j,com in enumerate(Compare)]
    a_conf_low_estimators = [Ref[j]-(Conf_bound_higher_beta*com) for j,com in enumerate(Compare)]
    a_conf_high_estimators = [Ref[j]-(Conf_bound_lower_beta*com) for j,com in enumerate(Compare)]
    # obtain the median values
    a = np.median(np.array(a_estimators))
    a_conf_low = np.median(np.array(a_conf_low_estimators))
    a_conf_high = np.median(np.array(a_conf_high_estimators))
    return a,a_conf_low,a_conf_high

def linearity_check_matrices(a,b,Compare,Ref):
    # Need to check whether or not approach is valid, using linearity testing 
    # (since it inherently assumes there is a linear relationship between the results of the two methods...)
    
    fittedvalues = a + b*Compare # obtain the fitted values
    higher = len(Ref[Ref>fittedvalues]) # number of values above the regression line
    lower = len(Ref[Ref<fittedvalues]) # number of values below the regression line
    #same = len(Ref[Ref==fittedvalues]) # number of values on the regression line --> should be equal to zero or very low!
    
    # set up a score matrix of the values, needed for the test!
    scorematrix = np.copy(Ref)
    for i in range(len(scorematrix)):
        if Ref[i] > fittedvalues[i]: # if above
            scorematrix[i] = np.sqrt(lower/higher) # assign this score
        elif Ref[i] < fittedvalues[i]: # if below
            scorematrix[i] = - np.sqrt(higher/lower) # assign this score
        else: # if on
            scorematrix[i] = 0. # assign this score
    
    # set up a distancematrix, consisting of the vertical distances (wrt to the regression line)
    # from data point to the regression line, needed for sorting the scores!
    distancematrix = np.array([(ref + ((1./b)*Compare[k]) - a)/np.sqrt(1. + (1./np.power(b,2.))) for k,ref in enumerate(Ref)])
    distance_indices = distancematrix.argsort() # obtain the indices needed for sorting
    # obtain the sorted distance and scorematrix
    sorted_distancematrix = distancematrix[distance_indices]
    sorted_scorematrix = scorematrix[distance_indices]
    return sorted_distancematrix,sorted_scorematrix,fittedvalues,higher,lower,distance_indices

def estimate_regression_confidence(a,a_conf_low,a_conf_high,b,Conf_bound_lower_beta,Conf_bound_higher_beta,Compare,eCompare,fittedvalues):
    low_sd_a = a-a_conf_low
    high_sd_a = a_conf_high-a
    low_sd_b = b-Conf_bound_lower_beta
    high_sd_b = Conf_bound_higher_beta-b
    low_bounds_sd = np.sqrt(np.power(low_sd_a,2)+np.power(Compare,2)*np.power(low_sd_b,2)+ np.power(b,2)*np.power(eCompare,2))
    high_bounds_sd = np.sqrt(np.power(high_sd_a,2)+np.power(Compare,2)*np.power(high_sd_b,2)+ np.power(b,2)*np.power(eCompare,2))
    low_bounds = fittedvalues - low_bounds_sd
    high_bounds = fittedvalues + high_bounds_sd
    return low_bounds,high_bounds

def plot_comparison_identity(Compare,fittedvalues,low_bounds,high_bounds,methodstring,interval_bootstrap,neginterval_bootstrap,methodstring2=False,nonextremal=False):
    sns.set_style('darkgrid')
    # obtain indices needed in order to properly display line plots!
    fittedsortindices = np.argsort(Compare)
    
    # 1st line plot shown in first P&B paper, comparison with identity line, and fitted/non-fitted
    plt.figure()
    plt.plot(Compare[fittedsortindices],Compare[fittedsortindices],label='Identity',ls='-')
    plt.plot(Compare[fittedsortindices],fittedvalues[fittedsortindices],label='Fit')
    plt.plot(Compare[fittedsortindices],low_bounds[fittedsortindices],c='k',ls='--',label='Conf. Bounds')
    plt.plot(Compare[fittedsortindices],high_bounds[fittedsortindices],c='k',ls='--')
    plt.plot(Compare[fittedsortindices],neginterval_bootstrap[fittedsortindices],c='m',ls='--',label='Conf. Bounds Bootstrap')
    plt.plot(Compare[fittedsortindices],interval_bootstrap[fittedsortindices],c='m',ls='--') 
    plt.xlabel(methodstring + ' ' + r'$\varpi$')
    if not methodstring2:
        plt.ylabel('Gaia ' + r'$\varpi$')
        if nonextremal:
            plt.title('Fitted Gaia ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$' + ' (non-extremal)')
        else:
            plt.title('Fitted Gaia ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$')
    else:
        plt.ylabel(methodstring2 + ' ' + r'$\varpi$')
        if nonextremal:
            plt.title('Fitted '+ methodstring2+ ' ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$'+ ' (non-extremal)')        
        else:
            plt.title('Fitted '+ methodstring2+ ' ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$')        
    Legend = plt.legend(frameon=True, fancybox=True, framealpha=1.0)
    frame = Legend.get_frame()
    frame.set_facecolor('White')
    return

def plot_cusum_statistic(absolute,x,y,higher,lower,sorted_colours,methodstring,methodstring2=False,nonextremal=False):
    # plot of the cusum statistic, which should be moderate in order to indicate linearity.
    markersize = 8 # change markersize
    if absolute:
        # absolute value of cusum statistic, with 99% (red), 95% (orange), and 90% (black) confidence intervals
        plt.figure()
        plt.scatter(x,np.absolute(y),label='|Cusum(i)|',color=[ color_dict[u] for u in sorted_colours],s=markersize)
        ax = plt.gca()
        first_legend = plt.legend(handles=patchList, loc=2, frameon=True, fancybox=True, framealpha=1.0)
        first_frame = first_legend.get_frame()
        first_frame.set_facecolor('White')
        ax.add_artist(first_legend)
        ax.axhline(y = 0. ,c='k', ls='--')
        ax.axhline(y = 1.63 * np.sqrt(higher+lower),label='99%',c='red')
        ax.axhline(y = 1.36 * np.sqrt(higher+lower),label='95%',c='yellow')
        ax.axhline(y = 1.22 * np.sqrt(higher+lower),label='90%',c='k')
        plt.xlabel('Rank i')
        plt.ylabel('|Cusum(i)|')
        if not methodstring2:
            if nonextremal:
                plt.title('Cusum statistic for P-B regression of GAIA ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$' + ' (non-extremal)')
            else:
                plt.title('Cusum statistic for P-B regression of GAIA ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$' )
        else:
            if nonextremal:
                plt.title('Cusum statistic for P-B regression of ' + methodstring + ' ' + r'$\varpi$' + ' vs ' + methodstring2 + ' ' + r'$\varpi$'+ ' (non-extremal)')
            else:
                plt.title('Cusum statistic for P-B regression of ' + methodstring + ' ' + r'$\varpi$' + ' vs ' + methodstring2 + ' ' + r'$\varpi$')
        Legend = plt.legend(loc=1, frameon=True, fancybox=True, framealpha=1.0)
        frame = Legend.get_frame()
        frame.set_facecolor('White')
    else:
        # true value of cusum statistic, with 99% (red), 95% (orange), and 90% (black) confidence intervals
        plt.figure()
        plt.scatter(x,y,label='Cusum(i)',color=[ color_dict[u] for u in sorted_colours],s=markersize)
        axe = plt.gca()
        second_legend = plt.legend(handles=patchList, loc=2, frameon=True, fancybox=True, framealpha=1.0)
        second_frame = second_legend.get_frame()
        second_frame.set_facecolor('White')
        axe.add_artist(second_legend)
        axe.axhline(y = 0. ,c='k', ls='--')
        axe.axhline(y = 1.63 * np.sqrt(higher+lower),label='99%',c='red')
        axe.axhline(y = 1.36 * np.sqrt(higher+lower),label='95%',c='yellow')
        axe.axhline(y = 1.22 * np.sqrt(higher+lower),label='90%',c='k')
        axe.axhline(y = -1.63 * np.sqrt(higher+lower),c='red')
        axe.axhline(y = -1.36 * np.sqrt(higher+lower),c='yellow')
        axe.axhline(y = -1.22 * np.sqrt(higher+lower),c='k')
        plt.xlabel('Rank i')
        plt.ylabel('Cusum(i)')
        if not methodstring2:
            if nonextremal:
                plt.title('Cusum statistic for P-B regression of GAIA ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$' + ' (non-extremal)')
            else:
                plt.title('Cusum statistic for P-B regression of GAIA ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$' )
        else:
            if nonextremal:
                plt.title('Cusum statistic for P-B regression of ' + methodstring2 + ' ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$'+ ' (non-extremal)')
            else:
                plt.title('Cusum statistic for P-B regression of ' + methodstring2 + ' ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$')
        Second_Legend = plt.legend(loc=1, frameon=True, fancybox=True, framealpha=1.0)
        frame = Second_Legend.get_frame()
        frame.set_facecolor('White')
    return

def plot_regression_residuals(x,Ref,fittedvalues,distance_indices,sorted_colours,methodstring,methodstring2=False,nonextremal=False):
    # figure containing regression residuals in terms of the same rank 
    plt.figure()
    axes = plt.gca()
    axes.axhline(y=0.,c='k',ls='--')
    third_legend = plt.legend(handles=patchList, loc=2, frameon=True, fancybox=True, framealpha=1.0)
    frame = third_legend.get_frame()
    frame.set_facecolor('White')
    axes.add_artist(third_legend)
    plt.scatter(x,Ref[distance_indices]-fittedvalues[distance_indices],marker='x',color=[ color_dict[u] for u in sorted_colours])
    plt.xlabel('Rank i')
    plt.ylabel('Residuals')
    if not methodstring2:
        if nonextremal:
            plt.title('Regression residuals for P-B regression of GAIA ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$' + ' (non-extremal)')
        else:
            plt.title('Regression residuals for P-B regression of GAIA ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$' )
    else:
        if nonextremal:
            plt.title('Regression residuals for P-B regression of ' + methodstring2 + ' ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$' + ' (non-extremal)')        
        else:
            plt.title('Regression residuals for P-B regression of ' + methodstring2 + ' ' + r'$\varpi$' + ' in function of ' + methodstring + ' ' + r'$\varpi$' )        
    return

#------------------------------------------------------------------------------
#           Bootstrapping Definitions (Based on Carpenter & Bithell 2000)
#------------------------------------------------------------------------------

big = True

def semi_param_resampling(Ref,Compare,a,b): # MODEL: Ref = b*Compare + a
    # y = g(beta,x) + residuals
    # estimate of beta = est(beta) = ^beta = [b,a]
    bootstraps = []
    residuals = Ref - ((b*Compare) + a*np.ones(len(Compare)))# observed - predicted
    mean_res = np.mean(residuals)
    adjusted_residuals = residuals - mean_res*np.ones(len(residuals))
    if big:
        B = 1999
    else:
        B = 999 
    for i in range(B):
        # resampling the adjusted residuals in a random way
        resampled_adjusted_residuals = resample(adjusted_residuals, n_samples=len(adjusted_residuals),random_state=i)
        new_estimates = (b*Compare) + (a*np.ones(len(Compare))) + resampled_adjusted_residuals
        bootstraps.append(new_estimates)
    return bootstraps,B

def calculate_bias(Ref,Compare,a,b,bootstraps,B):
    model_fits = (b*Compare) + a*np.ones(len(Compare))
    bias = []
    for r in range(len(model_fits)):
        test = model_fits[r]
        tester = []
        for t in range(len(bootstraps)):
            strapper = bootstraps[t]
            tester.append(strapper[r])
        tester = np.array(tester,dtype=np.float128)
        p = len(tester[tester<test])
        bias.append(norm.ppf(np.float(p)/np.float(B)))
    return np.array(bias)

def calculate_Q(B,bias):
    Z_95_percent = -1.960
    return (B+1)*norm.cdf(2.*bias - Z_95_percent)

interpolatedQ = True

def estimate_interval_pos(Ref,Compare,a,b,Q,B,bootstraps,interpolatedQ):
    interval = []
    integerQ = []
    for u in Q:
        integerQ.append(int(round(u)))
    integerQ = np.array(integerQ)
    model_fits = (b*Compare) + a*np.ones(len(Compare))
    for e in range(len(model_fits)):
        boots = []
        for d in range(len(bootstraps)):
            strappers = bootstraps[d]
            boots.append(strappers[e])
        boots = np.sort(np.array(boots))
        if interpolatedQ:
            z = math.floor(Q[e])
            r = z + 1
            if z > len(boots)-1:
                z -= 2
                r -= 2
            elif r > len(boots)-1:
                z -= 1 
                r -= 1
            invcdfz = norm.ppf(z/(B+1))
            invcdfr = norm.ppf(r/(B+1))
            invcdf = norm.ppf(Q[e]/(B+1))
            interval.append(boots[int(z)] + ((invcdf - invcdfz)/(invcdfr-invcdfz))*(boots[int(r)]-boots[int(z)]))
        else:
            if integerQ[e] > len(boots):
                interval.append(boots[len(boots)-1])
            else:
                interval.append(boots[integerQ[e]])
    interval = np.array(interval)
    delta = interval - model_fits
    neginterval = model_fits-delta
    return interval,neginterval
