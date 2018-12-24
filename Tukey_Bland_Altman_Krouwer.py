## coding: utf-8 
#!/usr/bin/env python2/3
#
# File: Tukey_Bland_Altman_Krouwer.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Module that implements the Tukey's mean difference plots/Bland-Altman plots/Krouwer plots 
# for the statistical comparison of the agreement of two methods.
#
# Publications of interest:
# Giavarina, Understanding Bland Altman analysis, Biochemica Medica, 25(2), pp.141-151, 2015
# http://www.biochemia-medica.com/en/journal/25/2/10.11613/BM.2015.015
# Bland & Altman, Statistical Methods for Assessing Agreement between two Methods, 327(8476), pp. 307-310, 1986
# http://www.sciencedirect.com/science/article/pii/S0140673686908378

# FOR NORMALITY & OUTLIER TESTS: see figure 4.7 in
# Ivezic et al., Statistics, Data Mining, and Machine Learning in Astronomy, Princeton University Press, 2014
# http://www.astroml.org/book_figures/chapter4/fig_anderson_darling.html

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
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy import odr
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
from patsy import dmatrices
import matplotlib.patches as mpatches
import sys
from uncertainties import unumpy

# Make a color dictionary, to be used during plotting
color_dict = { 'Blazhko':'orange', 'RRLyr':'blue'}
patchList = []
for key in color_dict:
        data_key = mpatches.Patch(color=color_dict[key], label=key)
        patchList.append(data_key)


def convert_uncert_df_to_nom_err(df):
    # generate dataframes containing the nominal values (nom) and the standard deviations (err) 
    # from the original dataframe containing ufloats
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

def f_ODR(B, x):
    # Linear function for orthogonal distance regression: y = m*x + b
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # Return an array in the same format as y, that is passed to Data.
    return B[0]*x + B[1]

def generate_fit_models(nom,err,x_variable,weighted,nonextremal=False):
    # Constructs the different linear fit models of the differences, used to determine biases.
    if nonextremal:
        # non-extremal differences: avoiding outliers!
        non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
        if len(non_extremal_indices) > 2:
            non_extremal_nom = nom.iloc[non_extremal_indices]
            non_extremal_err = err.iloc[non_extremal_indices]
            # generate design matrix for fit using Patsy
            non_extremal_y,non_extremal_X = dmatrices('Difference ~ ' + str(x_variable) , data=non_extremal_nom, return_type='dataframe')
            # generate a weighted or unweighted linear least squares model using Statsmodels
            if weighted:
                non_extremal_mod = sm.WLS(non_extremal_y,non_extremal_X,weights=1./(non_extremal_err['Difference']**2 + non_extremal_err[str(x_variable)]**2)**(1./2.))
            else:
                non_extremal_mod = sm.OLS(non_extremal_y,non_extremal_X)
                # generate a robust fitting model that punishes outliers, using the Huber loss function
            non_extremal_RLM_mod = sm.RLM(non_extremal_y,non_extremal_X,M=sm.robust.norms.HuberT())
            # generate data format for ODR
            non_extremal_ODR_data = odr.Data(non_extremal_nom[x_variable].values, non_extremal_nom['Difference'].values, 
                                wd=1./np.power(non_extremal_err[x_variable].values,2), 
                                we=1./np.power(non_extremal_err['Difference'].values,2))
            # generate an orthogonal distance fitting model (Deming regression)
            non_extremal_ODR_mod = odr.Model(f_ODR)
            # instantiate ODR with data, model and initial parameter estimate
            non_extremal_odr_mod = odr.ODR(non_extremal_ODR_data, non_extremal_ODR_mod, beta0=[0., 0.]) # initial estimate = 0 * X + 0 = no difference!
            # generate the fit results
            non_extremal_res = non_extremal_mod.fit()
            non_extremal_RLM_res = non_extremal_RLM_mod.fit()
            non_extremal_odr_res = non_extremal_odr_mod.run()
            # generate residuals and corresponding errors on the residual, to be used in ODR fit plot
            non_extremal_residual_odr,non_extremal_sigma_odr = calc_residual_sigma_odr(non_extremal_odr_res,non_extremal_nom[x_variable].values,
                                                             non_extremal_nom['Difference'].values,
                                                             non_extremal_err[x_variable].values,
                                                             non_extremal_err['Difference'].values)
            # generate the confidence bands of the (non-)weighted model
            non_extremal_prstd, non_extremal_iv_l, non_extremal_iv_u = wls_prediction_std(non_extremal_res)   


    # generate design matrix for fit using Patsy
    y,X = dmatrices('Difference ~ ' + str(x_variable) , data=nom, return_type='dataframe')
    # generate a weighted or unweighted linear least squares model using Statsmodels
    if weighted:
        mod = sm.WLS(y,X,weights=1./(err['Difference']**2 + err[str(x_variable)]**2)**(1./2.))
    else:
        mod = sm.OLS(y,X)
    # generate a robust fitting model that punishes outliers, using the Huber loss function
    RLM_mod = sm.RLM(y,X,M=sm.robust.norms.HuberT())
    # generate an orthogonal distance fitting model (Deming regression)
    ODR_mod = odr.Model(f_ODR)
    # generate data format for ODR
    ODR_data = odr.Data(nom[x_variable].values, nom['Difference'].values, 
                        wd=1./np.power(err[x_variable].values,2), 
                        we=1./np.power(err['Difference'].values,2))
    # instantiate ODR with data, model and initial parameter estimate
    odr_mod = odr.ODR(ODR_data, ODR_mod, beta0=[0., 0.]) # initial estimate = 0 * X + 0 = no difference!
    # generate the fit results
    res = mod.fit()
    RLM_res = RLM_mod.fit()
    odr_res = odr_mod.run()
    # generate residuals and corresponding errors on the residual, to be used in ODR fit plot
    residual_odr,sigma_odr = calc_residual_sigma_odr(odr_res,nom[x_variable].values,
                                                     nom['Difference'].values,
                                                     err[x_variable].values,
                                                     err['Difference'].values)
    # generate the confidence bands of the (non-)weighted model
    prstd, iv_l, iv_u = wls_prediction_std(res)  
    
    if nonextremal:
        if len(non_extremal_indices) > 2:
            return X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr,non_extremal_X,non_extremal_res,non_extremal_mod,non_extremal_RLM_res,non_extremal_RLM_mod,non_extremal_iv_u,non_extremal_iv_l,non_extremal_odr_res,non_extremal_residual_odr,non_extremal_sigma_odr
        else:
            return X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr           
    else:
        return X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr

def calc_residual_sigma_odr(output,x_data,y_data,x_sigma,y_sigma):
    # Calculate initial residuals and adjusted error 'sigma_odr'
    #                 for each data point
    p = output.beta
    delta   = output.delta   # estimated x-component of the residuals
    epsilon = output.eps     # estimated y-component of the residuals
    #    (dx_star,dy_star) are the projections of the errors of x and y onto
    #    the residual line, respectively, i.e. the differences in x & y
    #    between the data point and the point where the orthogonal residual
    #    line intersects the ellipse' created by x_sigma & y_sigma.
    dx_star = ( x_sigma*np.sqrt( ((y_sigma*delta)**2) /
                    ( (y_sigma*delta)**2 + (x_sigma*epsilon)**2 ) ) )
    dy_star = ( y_sigma*np.sqrt( ((x_sigma*epsilon)**2) /
                    ( (y_sigma*delta)**2 + (x_sigma*epsilon)**2 ) ) )
    # calculate the 'total projected uncertainty' based on these projections
    sigma_odr = np.sqrt(dx_star**2 + dy_star**2)
    # residual is positive if the point lies above the fitted curve,
    #             negative if below
    residual_odr = ( np.sign(y_data-f_ODR(p,x_data))
                  * np.sqrt(delta**2 + epsilon**2) )
    return residual_odr,sigma_odr

def Bland_Altman_main(plx_df,method1,method2,our_script,outfile,weighted = True, percent=False, logplot=False):
    # loads in the parallax dataframe in order to obtain parameters for
    # a Tukey mean-difference plot (also called a Bland-Altman plot), or Krouwer plot,
    # in order to better compare the different methods
    
    # generate list of stars  
    stars = list(plx_df)
    # generate list of methods
    methods = list(plx_df.index)
    
    # if using our dereddening script as well
    if our_script:
        # double PML method comparison = 3 + 3 + 1 + 1 rows
        if len(plx_df) > 6:
            with open(outfile, 'w') as f:
                for i in range(0,6):
                    for j in range(i+1,6):
                        # obtain the names of the different methods (different dereddening methods, for the different PML relations)
                        methodname1 = methods[i]
                        methodname2 = methods[j]
                        # beta = differences of parallaxes, alfa = means of parallaxes, GAIA = gaia parallaxes
                        beta = pd.DataFrame(plx_df.iloc[i].values - plx_df.iloc[j].values,index=stars,columns=['Difference']) # method 1 - method 2
                        alfa = pd.DataFrame((plx_df.iloc[i].values + plx_df.iloc[j].values)/2.,index=stars,columns=['Mean'])
                        GAIA = pd.DataFrame(plx_df.loc['GAIA'].values, index=stars,columns=['Reference'])
                        # concatenate the different dataframes into one encompassing dataframe
                        plot_df = pd.concat([beta,alfa,GAIA],axis='columns')
                        if re.search('(Sesar)',methods[i]):
                            # select only RRab stars, since Sesar relation only applies to those
                            plt_df = plot_df[plx_df.loc['RRAB/RRC']=='RRAB']
                            plt_alfa = alfa[plx_df.loc['RRAB/RRC']=='RRAB']
                            plt_beta = beta[plx_df.loc['RRAB/RRC']=='RRAB']
                            plt_plx_df = plx_df.loc[:,plx_df.loc['RRAB/RRC']=='RRAB']
                            # convert df input containing ufloats to readable output
                            nom,err = convert_uncert_df_to_nom_err(plt_df)
                            non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                            if len(non_extremal_indices) > 2:
                                # Do the actual fitting
                                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr,non_extremal_X,non_extremal_res,non_extremal_mod,non_extremal_RLM_res,non_extremal_RLM_mod,non_extremal_iv_u,non_extremal_iv_l,non_extremal_odr_res,non_extremal_residual_odr,non_extremal_sigma_odr = generate_fit_models(nom,err,'Mean',weighted,nonextremal=True)
                                # Generate the Bland-Altman/Krouwer plots
                                Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                          odr_res,sigma_odr,residual_odr,methodname1,
                                                          method2=methodname2,percent=percent,logplot=logplot)
                                Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,non_extremal_X,
                                                          non_extremal_res,non_extremal_iv_u,non_extremal_iv_l,non_extremal_RLM_res,
                                                          non_extremal_odr_res,non_extremal_sigma_odr,non_extremal_residual_odr,methodname1,
                                                          method2=methodname2,percent=percent,logplot=logplot,nonextremal=True)
                            else:
                                # Do the actual fitting
                                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                                # Generate the Bland-Altman/Krouwer plots
                                Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                          odr_res,sigma_odr,residual_odr,methodname1,
                                                          method2=methodname2,percent=percent,logplot=logplot)
                            # Generate normality histograms
                            Normality_histogram(nom['Difference'],methodname1,method2=methodname2)
                            # Generate normality assessments
                            Normality_tests(f,nom['Difference'],methodname1)
                            Normality_tests(f,nom['Difference'],methodname2)
                            # Generate regression diagnostics plots
                            regression_diagnostics_plot(nom,mod,X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey')
                            if len(non_extremal_indices) > 2:
                                regression_diagnostics_plot(nom,non_extremal_mod,non_extremal_X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey',non_extremal=True)
                        else:
                            # convert df input containing ufloats to readable output
                            nom,err = convert_uncert_df_to_nom_err(plot_df)
                            non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                            if len(non_extremal_indices) > 2:
                                # Do the actual fitting
                                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr,non_extremal_X,non_extremal_res,non_extremal_mod,non_extremal_RLM_res,non_extremal_RLM_mod,non_extremal_iv_u,non_extremal_iv_l,non_extremal_odr_res,non_extremal_residual_odr,non_extremal_sigma_odr = generate_fit_models(nom,err,'Mean',weighted,nonextremal=True)
                                # Generate the Bland-Altman/Krouwer plots
                                Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                          odr_res,sigma_odr,residual_odr,methodname1,
                                                          method2=methodname2,percent=percent,logplot=logplot)
                                Bland_Altman_Krouwer_plot(f,plx_df,nom,err,non_extremal_X,
                                                          non_extremal_res,non_extremal_iv_u,non_extremal_iv_l,non_extremal_RLM_res,
                                                          non_extremal_odr_res,non_extremal_sigma_odr,non_extremal_residual_odr,methodname1,
                                                          method2=methodname2,percent=percent,logplot=logplot,nonextremal=True)
                            else:
                                # Do the actual fitting
                                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                                # Generate the Bland-Altman/Krouwer plots
                                Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                          odr_res,sigma_odr,residual_odr,methodname1,
                                                          method2=methodname2,percent=percent,logplot=logplot)
                            # Generate normality histograms
                            Normality_histogram(nom['Difference'],methodname1,method2=methodname2)
                            # Generate normality assessments
                            Normality_tests(f,nom['Difference'],methodname1)
                            Normality_tests(f,nom['Difference'],methodname2)
                            # Generate regression diagnostics plots
                            regression_diagnostics_plot(nom,mod,X,alfa,beta,plx_df,'Mean',weighted,'Tukey')
                            if len(non_extremal_indices) > 2:
                                regression_diagnostics_plot(nom,non_extremal_mod,non_extremal_X,alfa,beta,plx_df,'Mean',weighted,'Tukey',non_extremal=True)
            return
    
        # PML + GAIA method comparison = 3 + 1 + 1 + 1 rows
        else:
            with open(outfile, 'w') as f:
                for i in range(3):
                    # obtain method name (distinction between dereddening)
                    methodname = methods[i]
                    # beta = differences of parallaxes, alfa = means of parallaxes, alfa_Krouwer = gaia parallax
                    beta = pd.DataFrame(plx_df.loc['GAIA'].values - plx_df.iloc[i].values, index=stars,columns=['Difference']) # GAIA - PML parallax
                    alfa_Krouwer = pd.DataFrame(plx_df.loc['GAIA'].values, index=stars,columns=['Reference'])
                    alfa = pd.DataFrame((plx_df.loc['GAIA'].values + plx_df.iloc[i].values)/2., index=stars,columns=['Mean'])
                    # concatenate the different dataframes into one encompassing dataframe
                    plot_df = pd.concat([beta,alfa_Krouwer,alfa],axis='columns')
                    if re.search('(Sesar)',methods[i]):
                        # select only RRab stars, since Sesar relation only applies to those
                        plt_df = plot_df[plx_df.loc['RRAB/RRC']=='RRAB']
                        plt_alfa = alfa[plx_df.loc['RRAB/RRC']=='RRAB']
                        plt_beta = beta[plx_df.loc['RRAB/RRC']=='RRAB']
                        plt_plx_df = plx_df.loc[:,plx_df.loc['RRAB/RRC']=='RRAB']
                        plt_alfa_Krouwer = alfa_Krouwer[plx_df.loc['RRAB/RRC']=='RRAB']
                        # convert df input containing ufloats to readable output
                        nom,err = convert_uncert_df_to_nom_err(plt_df)                        
                        non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                        if len(non_extremal_indices) > 2:
                            # Do the actual fitting
                            X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr,non_extremal_X,non_extremal_res,non_extremal_mod,non_extremal_RLM_res,non_extremal_RLM_mod,non_extremal_iv_u,non_extremal_iv_l,non_extremal_odr_res,non_extremal_residual_odr,non_extremal_sigma_odr = generate_fit_models(nom,err,'Mean',weighted,nonextremal=True) # Tukey
                            X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K,non_extremal_X_K,non_extremal_res_K,non_extremal_mod_K,non_extremal_RLM_res_K,non_extremal_RLM_mod_K,non_extremal_iv_u_K,non_extremal_iv_l_K,non_extremal_odr_res_K,non_extremal_residual_odr_K,non_extremal_sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted,nonextremal=True) # Krouwer
                            # Generate the Bland-Altman/Krouwer plots
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                                      RLM_res_K,odr_res_K,sigma_odr_K,
                                                      residual_odr_K,methodname,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                      odr_res,sigma_odr,residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,non_extremal_X_K,non_extremal_res_K,non_extremal_iv_u_K,non_extremal_iv_l_K,
                                                      non_extremal_RLM_res_K,non_extremal_odr_res_K,non_extremal_sigma_odr_K,
                                                      non_extremal_residual_odr_K,methodname,percent=percent,logplot=logplot,nonextremal=True)
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,non_extremal_X,
                                                      non_extremal_res,non_extremal_iv_u,non_extremal_iv_l,non_extremal_RLM_res,
                                                      non_extremal_odr_res,non_extremal_sigma_odr,non_extremal_residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot,nonextremal=True)
                        else:
                            # Do the actual fitting
                            X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted) # Tukey
                            X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted) # Krouwer
                            # Generate the Bland-Altman/Krouwer plots
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                                      RLM_res_K,odr_res_K,sigma_odr_K,
                                                      residual_odr_K,methodname,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                      odr_res,sigma_odr,residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot)
                        # Generate normality histograms
                        Normality_histogram(nom['Difference'],methodname,method2='GAIA Reference')
                        # Generate normality assessments
                        Normality_tests(f,nom['Difference'],methodname)
                        # Generate regression diagnostics plots
                        regression_diagnostics_plot(nom,mod,X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey')
                        regression_diagnostics_plot(nom,mod_K,X_K,plt_alfa_Krouwer,plt_beta,plt_plx_df,'Reference',weighted,'Krouwer')
                        if len(non_extremal_indices) > 2:
                            regression_diagnostics_plot(nom,non_extremal_mod,non_extremal_X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey',non_extremal=True)
                            regression_diagnostics_plot(nom,non_extremal_mod_K,non_extremal_X_K,plt_alfa_Krouwer,plt_beta,plt_plx_df,'Reference',weighted,'Krouwer',non_extremal=True)
                    else:
                        # convert df input containing ufloats to readable output
                        nom,err = convert_uncert_df_to_nom_err(plot_df)
                        non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                        if len(non_extremal_indices) > 2:
                            # Do the actual fitting
                            X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr,non_extremal_X,non_extremal_res,non_extremal_mod,non_extremal_RLM_res,non_extremal_RLM_mod,non_extremal_iv_u,non_extremal_iv_l,non_extremal_odr_res,non_extremal_residual_odr,non_extremal_sigma_odr = generate_fit_models(nom,err,'Mean',weighted,nonextremal=True) # Tukey
                            X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K,non_extremal_X_K,non_extremal_res_K,non_extremal_mod_K,non_extremal_RLM_res_K,non_extremal_RLM_mod_K,non_extremal_iv_u_K,non_extremal_iv_l_K,non_extremal_odr_res_K,non_extremal_residual_odr_K,non_extremal_sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted,nonextremal=True) # Krouwer
                            # Generate the Bland-Altman/Krouwer plots
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                                      RLM_res_K,odr_res_K,sigma_odr_K,
                                                      residual_odr_K,methodname,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                      odr_res,sigma_odr,residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,non_extremal_X_K,non_extremal_res_K,
                                                      non_extremal_iv_u_K,non_extremal_iv_l_K,
                                                      non_extremal_RLM_res_K,non_extremal_odr_res_K,non_extremal_sigma_odr_K,
                                                      non_extremal_residual_odr_K,methodname,percent=percent,logplot=logplot,nonextremal=True)
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,non_extremal_X,
                                                      non_extremal_res,non_extremal_iv_u,non_extremal_iv_l,non_extremal_RLM_res,
                                                      non_extremal_odr_res,non_extremal_sigma_odr,non_extremal_residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot,nonextremal=True)
 
                        else:
                            # Do the actual fitting
                            X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted) # Tukey
                            X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted) # Krouwer
                            # Generate the Bland-Altman/Krouwer plots
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                                      RLM_res_K,odr_res_K,sigma_odr_K,
                                                      residual_odr_K,methodname,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                      odr_res,sigma_odr,residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot)
 
                        # Generate normality histograms
                        Normality_histogram(nom['Difference'],methodname,method2='GAIA Reference')
                        # Generate normality assessments
                        Normality_tests(f,nom['Difference'],methodname)
                        # Generate regression diagnostics plots
                        regression_diagnostics_plot(nom,mod,X,alfa,beta,plx_df,'Mean',weighted,'Tukey')
                        regression_diagnostics_plot(nom,mod_K,X_K,alfa_Krouwer,beta,plx_df,'Reference',weighted,'Krouwer')
                        if len(non_extremal_indices) > 2:
                            regression_diagnostics_plot(nom,non_extremal_mod,non_extremal_X,alfa,beta,plx_df,'Mean',weighted,'Tukey',non_extremal=True)
                            regression_diagnostics_plot(nom,non_extremal_mod_K,non_extremal_X_K,alfa_Krouwer,beta,plx_df,'Reference',weighted,'Krouwer',non_extremal=True)
            return
    else:
        # double PML method comparison = 3 + 3 + 1 + 1 rows
        if len(plx_df) > 5:
            with open(outfile, 'w') as f:
                for i in range(0,4):
                    for j in range(i+1,4):
                        # obtain the names of the different methods (different dereddening methods, for the different PML relations)
                        methodname1 = methods[i]
                        methodname2 = methods[j]
                        # beta = differences of parallaxes, alfa = means of parallaxes, GAIA = gaia parallaxes
                        beta = pd.DataFrame(plx_df.iloc[i].values - plx_df.iloc[j].values,index=stars,columns=['Difference']) # method 1 - method 2
                        alfa = pd.DataFrame((plx_df.iloc[i].values + plx_df.iloc[j].values)/2.,index=stars,columns=['Mean'])
                        GAIA = pd.DataFrame(plx_df.loc['GAIA'].values, index=stars,columns=['Reference'])
                        # concatenate the different dataframes into one encompassing dataframe
                        plot_df = pd.concat([beta,alfa,GAIA],axis='columns')
                        if re.search('(Sesar)',methods[i]):
                            # select only RRab stars, since Sesar relation only applies to those
                            plt_df = plot_df[plx_df.loc['RRAB/RRC']=='RRAB']
                            plt_alfa = alfa[plx_df.loc['RRAB/RRC']=='RRAB']
                            plt_beta = beta[plx_df.loc['RRAB/RRC']=='RRAB']
                            plt_plx_df = plx_df.loc[:,plx_df.loc['RRAB/RRC']=='RRAB']
                            # convert df input containing ufloats to readable output
                            nom,err = convert_uncert_df_to_nom_err(plt_df)
                            non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                            if len(non_extremal_indices) > 2:
                                # Do the actual fitting
                                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr,non_extremal_X,non_extremal_res,non_extremal_mod,non_extremal_RLM_res,non_extremal_RLM_mod,non_extremal_iv_u,non_extremal_iv_l,non_extremal_odr_res,non_extremal_residual_odr,non_extremal_sigma_odr = generate_fit_models(nom,err,'Mean',weighted,nonextremal=True)
                                # Generate the Bland-Altman/Krouwer plots
                                Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                              odr_res,sigma_odr,residual_odr,methodname1,
                                                              method2=methodname2,percent=percent,logplot=logplot)
                                Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,non_extremal_X,
                                                              non_extremal_res,non_extremal_iv_u,non_extremal_iv_l,non_extremal_RLM_res,
                                                              non_extremal_odr_res,non_extremal_sigma_odr,non_extremal_residual_odr,methodname1,
                                                              method2=methodname2,percent=percent,logplot=logplot,nonextremal=True)
                            else:
                                # Do the actual fitting
                                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                                # Generate the Bland-Altman/Krouwer plots
                                Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                              odr_res,sigma_odr,residual_odr,methodname1,
                                                              method2=methodname2,percent=percent,logplot=logplot)
 
                            # Generate normality histograms
                            Normality_histogram(nom['Difference'],methodname1,method2=methodname2)
                            # Generate normality assessments
                            Normality_tests(f,nom['Difference'],methodname1)
                            Normality_tests(f,nom['Difference'],methodname2)
                            # Generate regression diagnostics plots
                            regression_diagnostics_plot(nom,mod,X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey')
                            if len(non_extremal_indices) > 2:                            
                                regression_diagnostics_plot(nom,non_extremal_mod,non_extremal_X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey',non_extremal=True)
                        else:
                            # convert df input containing ufloats to readable output
                            nom,err = convert_uncert_df_to_nom_err(plot_df)
                            non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                            if len(non_extremal_indices) > 2:
                                # Do the actual fitting
                                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr,non_extremal_X,non_extremal_res,non_extremal_mod,non_extremal_RLM_res,non_extremal_RLM_mod,non_extremal_iv_u,non_extremal_iv_l,non_extremal_odr_res,non_extremal_residual_odr,non_extremal_sigma_odr = generate_fit_models(nom,err,'Mean',weighted,nonextremal=True)
                                # Generate the Bland-Altman/Krouwer plots
                                Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                              odr_res,sigma_odr,residual_odr,methodname1,
                                                              method2=methodname2,percent=percent,logplot=logplot)
                                Bland_Altman_Krouwer_plot(f,plx_df,nom,err,non_extremal_X,
                                                              non_extremal_res,non_extremal_iv_u,non_extremal_iv_l,non_extremal_RLM_res,
                                                              non_extremal_odr_res,non_extremal_sigma_odr,non_extremal_residual_odr,methodname1,
                                                              method2=methodname2,percent=percent,logplot=logplot,nonextremal=True)
                            else:
                                # Do the actual fitting
                                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                                # Generate the Bland-Altman/Krouwer plots
                                Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                                              odr_res,sigma_odr,residual_odr,methodname1,
                                                              method2=methodname2,percent=percent,logplot=logplot)
                            # Generate normality histograms
                            Normality_histogram(nom['Difference'],methodname1,method2=methodname2)
                            # Generate normality assessments
                            Normality_tests(f,nom['Difference'],methodname1)
                            Normality_tests(f,nom['Difference'],methodname2)
                            # Generate regression diagnostics plots
                            regression_diagnostics_plot(nom,mod,X,alfa,beta,plx_df,'Mean',weighted,'Tukey')
                            if len(non_extremal_indices) > 2:                            
                                regression_diagnostics_plot(nom,non_extremal_mod,non_extremal_X,alfa,beta,plx_df,'Mean',weighted,'Tukey',non_extremal=True)
            return

        # PML + GAIA method comparison = 3 + 1 + 1 + 1 rows
        else:
            with open(outfile, 'w') as f:
                for i in range(2):
                    # obtain method name (distinction between dereddening)
                    methodname = methods[i]                  
                    # beta = differences of parallaxes, alfa = means of parallaxes, alfa_Krouwer = gaia parallax
                    beta = pd.DataFrame(plx_df.loc['GAIA'].values - plx_df.iloc[i].values, index=stars,columns=['Difference']) # GAIA - PML
                    alfa_Krouwer = pd.DataFrame(plx_df.loc['GAIA'].values, index=stars,columns=['Reference'])
                    alfa = pd.DataFrame((plx_df.loc['GAIA'].values + plx_df.iloc[i].values)/2., index=stars,columns=['Mean'])
                    # concatenate the different dataframes into one encompassing dataframe
                    plot_df = pd.concat([beta,alfa_Krouwer,alfa],axis='columns')
                    if re.search('(Sesar)',methods[i]):
                        # select only RRab stars, since Sesar relation only applies to those
                        plt_df = plot_df[plx_df.loc['RRAB/RRC']=='RRAB']
                        plt_alfa = alfa[plx_df.loc['RRAB/RRC']=='RRAB']
                        plt_beta = beta[plx_df.loc['RRAB/RRC']=='RRAB']
                        plt_plx_df = plx_df.loc[:,plx_df.loc['RRAB/RRC']=='RRAB']
                        plt_alfa_Krouwer = alfa_Krouwer[plx_df.loc['RRAB/RRC']=='RRAB']
                        # convert df input containing ufloats to readable output
                        nom,err = convert_uncert_df_to_nom_err(plt_df)
                        non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                        if len(non_extremal_indices) > 2:
                            # Do the actual fitting
                            X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr,non_extremal_X,non_extremal_res,non_extremal_mod,non_extremal_RLM_res,non_extremal_RLM_mod,non_extremal_iv_u,non_extremal_iv_l,non_extremal_odr_res,non_extremal_residual_odr,non_extremal_sigma_odr = generate_fit_models(nom,err,'Mean',weighted,nonextremal=True)
                            X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K,non_extremal_X_K,non_extremal_res_K,non_extremal_mod_K,non_extremal_RLM_res_K,non_extremal_RLM_mod_K,non_extremal_iv_u_K,non_extremal_iv_l_K,non_extremal_odr_res_K,non_extremal_residual_odr_K,non_extremal_sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted,nonextremal=True)
                            # Generate the Bland-Altman/Krouwer plots
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                                          RLM_res_K,odr_res_K,sigma_odr_K,
                                                          residual_odr_K,methodname,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,odr_res,sigma_odr,residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,non_extremal_X_K,non_extremal_res_K,non_extremal_iv_u_K,non_extremal_iv_l_K,
                                                          non_extremal_RLM_res_K,non_extremal_odr_res_K,non_extremal_sigma_odr_K,
                                                          non_extremal_residual_odr_K,methodname,percent=percent,logplot=logplot,nonextremal=True)
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,non_extremal_X,non_extremal_res,non_extremal_iv_u,non_extremal_iv_l,
                                                      non_extremal_RLM_res,non_extremal_odr_res,non_extremal_sigma_odr,non_extremal_residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot,nonextremal=True)
                        else:   
                            # Do the actual fitting
                            X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                            X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted)
                            # Generate the Bland-Altman/Krouwer plots
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                                          RLM_res_K,odr_res_K,sigma_odr_K,
                                                          residual_odr_K,methodname,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,odr_res,sigma_odr,residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot)

                        # Generate normality histograms
                        Normality_histogram(nom['Difference'],methodname,method2='GAIA Reference')
                        # Generate normality assessments
                        Normality_tests(f,nom['Difference'],methodname)
                        # Generate regression diagnostics plots
                        regression_diagnostics_plot(nom,mod,X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey')
                        regression_diagnostics_plot(nom,mod_K,X_K,plt_alfa_Krouwer,plt_beta,plt_plx_df,'Reference',weighted,'Krouwer')
                        if len(non_extremal_indices) > 2:
                            regression_diagnostics_plot(nom,non_extremal_mod,non_extremal_X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey',non_extremal=True)
                            regression_diagnostics_plot(nom,non_extremal_mod_K,non_extremal_X_K,plt_alfa_Krouwer,plt_beta,plt_plx_df,'Reference',weighted,'Krouwer',non_extremal=True)
                    else:
                        # convert df input containing ufloats to readable output
                        nom,err = convert_uncert_df_to_nom_err(plot_df)
                        non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
                        if len(non_extremal_indices) > 2:
                            # Do the actual fitting
                            X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr,non_extremal_X,non_extremal_res,non_extremal_mod,non_extremal_RLM_res,non_extremal_RLM_mod,non_extremal_iv_u,non_extremal_iv_l,non_extremal_odr_res,non_extremal_residual_odr,non_extremal_sigma_odr = generate_fit_models(nom,err,'Mean',weighted,nonextremal=True)
                            X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K,non_extremal_X_K,non_extremal_res_K,non_extremal_mod_K,non_extremal_RLM_res_K,non_extremal_RLM_mod_K,non_extremal_iv_u_K,non_extremal_iv_l_K,non_extremal_odr_res_K,non_extremal_residual_odr_K,non_extremal_sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted,nonextremal=True)
                            # Generate the Bland-Altman/Krouwer plots
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                                          RLM_res_K,odr_res_K,sigma_odr_K,
                                                          residual_odr_K,
                                                          methodname,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,odr_res,sigma_odr,residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,non_extremal_X_K,non_extremal_res_K,non_extremal_iv_u_K,non_extremal_iv_l_K,
                                                          non_extremal_RLM_res_K,non_extremal_odr_res_K,non_extremal_sigma_odr_K,
                                                          non_extremal_residual_odr_K,
                                                          methodname,percent=percent,logplot=logplot,nonextremal=True)
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,non_extremal_X,non_extremal_res,non_extremal_iv_u,non_extremal_iv_l,
                                                      non_extremal_RLM_res,non_extremal_odr_res,non_extremal_sigma_odr,non_extremal_residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot,nonextremal=True)

                        else:
                            # Do the actual fitting
                            X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                            X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted)
                            # Generate the Bland-Altman/Krouwer plots
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                                          RLM_res_K,odr_res_K,sigma_odr_K,
                                                          residual_odr_K,
                                                          methodname,percent=percent,logplot=logplot)
                            Bland_Altman_Krouwer_plot(f,plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,odr_res,sigma_odr,residual_odr,
                                                      methodname,method2=method2,percent=percent,logplot=logplot)
                        # Generate normality histograms
                        Normality_histogram(nom['Difference'],methodname,method2='GAIA Reference')
                        # Generate normality assessments
                        Normality_tests(f,nom['Difference'],methodname)
                        # Generate regression diagnostics plots
                        regression_diagnostics_plot(nom,mod,X,alfa,beta,plx_df,'Mean',weighted,'Tukey')
                        regression_diagnostics_plot(nom,mod_K,X_K,alfa_Krouwer,beta,plx_df,'Reference',weighted,'Krouwer')
                        if len(non_extremal_indices) > 2:
                            regression_diagnostics_plot(nom,non_extremal_mod,non_extremal_X,alfa,beta,plx_df,'Mean',weighted,'Tukey',non_extremal=True)
                            regression_diagnostics_plot(nom,non_extremal_mod_K,non_extremal_X_K,alfa_Krouwer,beta,plx_df,'Reference',weighted,'Krouwer',non_extremal=True)
            return

 

def regression_diagnostics_plot(nom,mod,X,plotalfa,plotbeta,df,x_string,weighted,plotstring,non_extremal=False):
    if non_extremal:
        # select stars that do no display extremal differences
        non_extremal_indices = np.argwhere((nom['Difference'].values < 1.) & (nom['Difference'].values > -1.)).flatten()
        sortedstars = nom.index.values
        plotalfa = plotalfa.reindex(sortedstars) # make sure df is ordered in same way as nom
        plotbeta = plotbeta.reindex(sortedstars) # make sure df is ordered in same way as nom
        plotalfa = plotalfa.copy().iloc[non_extremal_indices]
        plotbeta = plotbeta.copy().iloc[non_extremal_indices]
        df = df.reindex(columns=sortedstars) # make sure df is ordered in same way as nom
        df = df.copy().T.iloc[non_extremal_indices].T
    
    if len(plotalfa) > 30:
        markersize = 12 # change markersize
    elif len(plotalfa) > 50:
        markersize = 8 # change markersize
    else:
        markersize= 36 # default        
    # get nominal values
    alfa,_ = convert_uncert_df_to_nom_err(plotalfa)
    beta,_ = convert_uncert_df_to_nom_err(plotbeta)
    # if weighted fit, convert the fitted weighted parameters to a format that is readible for the diagnostics plot tools (so that one displays the diagnostics for the weighted parameters)
    if weighted:
        res_infl_mod = sm.OLS(pd.DataFrame(mod.wendog,index=list(alfa.index),columns=['Difference']),
                              pd.DataFrame(mod.wexog,index=list(alfa.index),columns=['Intercept',x_string]))
        res_infl = res_infl_mod.fit()
    else:
        res_infl = mod.fit()
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    # Generate Cook's distance influence plot
    sm.graphics.influence_plot(res_infl, ax=axes[0,0], criterion="cooks") 
    # Generate leverage plot
    sm.graphics.plot_leverage_resid2(res_infl, ax=axes[0,1]) 
    # Generate fit plot
    sm.graphics.plot_fit(res_infl, x_string, ax=axes[1,0]) 
    # Generate residual plot  
    if len(plotalfa) > 30:
        axes[1,1].scatter(alfa.T,beta.T - res_infl.predict(X).T, color=[ color_dict[u] for u in df.loc['Blazhko/RRLyr'] ],s=markersize) 
    else:
        axes[1,1].scatter(alfa.T,beta.T - res_infl.predict(X).T, color=[ color_dict[u] for u in df.loc['Blazhko/RRLyr'] ]) 
    axes[1,1].axhline(color='r')
    axes[1,1].set_xlabel(x_string + " " + r'$\varpi$' + " (mas)")
    axes[1,1].set_ylabel("Residual " + r'$\varpi$' + " (mas)")
    if non_extremal:
        axes[1,1].set_title("Residual Plot (" + plotstring + ", non-extremal)")
    else:
        axes[1,1].set_title("Residual Plot (" + plotstring + ")")
    # Generate extra legend displaying whether the RR Lyrae star is Blazhko-modulated or not
    third_legend = plt.legend(handles=patchList, frameon=True, fancybox=True, framealpha=1.0)
    third_frame = third_legend.get_frame()
    third_frame.set_facecolor('White')
    axes[1,1].add_artist(third_legend)
    return

def Normality_histogram(differences,method1,method2='GAIA Reference'):
    # create a histogram of the differences, containing a Kernel Density Estimation and normal distribution fit,
    # in order to assess normality of the differences distribution.
    # (Need normality in this parameter in order to use the prediction/confidence intervals, as well as limits of agreement! See Bland-Altman publications mentioned on top.)
    sns.set_style('darkgrid')
    plt.figure()
    sns.distplot(differences,norm_hist=True) # histogram + Kernel Density estimation ("KDE")
    sns.distplot(differences,fit=norm,kde=False,norm_hist=True) # histogram (overlay) + normal distribution fit ("Norm")
    # plot an estimate of the normal distribution without extremal values 
    # (as the distribution fit is affected significantly by those)
    non_extremal_indices = np.argwhere((differences < 1.) & (differences > -1.)).flatten() # exclude any differences above 1.5 or below -1.5
    if len(non_extremal_indices) > 2:
        sns.distplot(differences[non_extremal_indices],fit=norm,kde=False,hist=False,fit_kws={"color":"red"},norm_hist=True)
        Legend = plt.legend(['Norm','Norm non-extremal', 'KDE'],loc='upper left', frameon=True, fancybox=True, framealpha=1.0)
    else:
        Legend = plt.legend(['Norm', 'KDE'],loc='upper left', frameon=True, fancybox=True, framealpha=1.0)        
    frame = Legend.get_frame()
    frame.set_facecolor('White')
    if method2=='GAIA Reference':
        plt.xlabel( method2 + ' ' + r'$\varpi$' +  ' - ' + method1 + ' ' + r'$\varpi$')
    else:
        plt.xlabel( method1 + ' ' + r'$\varpi$' +  ' - ' + method2 + ' ' + r'$\varpi$')
    plt.ylabel('Probability Density')
    return

def Pair_grid(df_list,df,GAIA):
    if len(df_list[0]) > 30:
        markersize = 12 # change markersize
    elif len(df_list[0]) > 50:
        markersize = 8 # change markersize
    else:
        markersize= 36 # default  
    # use first item/method in dataframe list to obtain the star list
    totdf = df_list[0]
    # create dataframe containing GAIA data
    GAIAdf = pd.DataFrame(GAIA.reshape(1,len(GAIA)),index=["GAIA"],columns=list(totdf))
    # append results of other methods
    for i in range(1,len(df_list)):
        totdf = pd.concat([totdf,df_list[i]])
    # construct the final dataframe containing results of all methods and GAIA
    totdf = pd.concat([totdf,GAIAdf])
    # obtaining partial dataframes consisting of results using different dereddening methods
    script = totdf[~totdf.index.to_series().str.contains('(_SFD)|(_S_F)')] # our dereddening script
    S_F = pd.concat([totdf[totdf.index.to_series().str.contains('(_SFD)')],GAIAdf]) # S_F dust map (have to add GAIA data again)
    SFD = pd.concat([totdf[totdf.index.to_series().str.contains('(_S_F)')],GAIAdf]) # SFD dust map (have to add GAIA data again)
    # Loop over the results list using different dereddening methods, generating the pair grid for each
    for j in [script,S_F,SFD]:
        # get the nominal values of the dataframe containing ufloats
        nom,_ = convert_uncert_df_to_nom_err(j)
        # generate necessary dataframes for pair grid
        sns.set_context("talk")
        transpose_nom = nom.T
        nom_na = transpose_nom.dropna(axis=1)
        transpose_nom['B/RR'] = pd.Series(df.loc['B/RR'].values, index=transpose_nom.index) # add Blazhko/RRLyr distinction
        transpose_nom_nan = transpose_nom.dropna(axis=1) # drop any columns containing NaN values 
        # (i.e. in our case: Sesar relation column, since this is only applicable to RRAB, in principle, if we find other RRAB only relations we can compare those with Sesar!)

        # generate the actual pair grid (for more detailed information, see Seaborn website)
        g = sns.PairGrid(nom_na)
        g.map_upper(plt.scatter,color=[ color_dict[u] for u in transpose_nom_nan['B/RR'] ],s=markersize) # scatterplots making a coloured distinction between Blazhko and RRLyr
        g.map_lower(sns.kdeplot) # 2D KDE
        g.map_diag(sns.kdeplot, lw=2, legend=False); # 1D KDE on diagonals
        g.add_legend(handles=patchList) # legend containing the Blazkho/RRLyr distinction       
    return

def Normality_tests(outfile,differences,method): 
    # testing the non-extremal distribution
    non_extremal_indices = np.argwhere((differences < 1.) & (differences > -1.)).flatten()
    if len(non_extremal_indices) > 2:
        non_extremal_differences = differences[non_extremal_indices]
        # catastrophic outlier ratio calculation (see Ivezic et al. (2014)):
        stdev_differences_non_extremal = np.std(non_extremal_differences, ddof=1)
        iqr_non_extremal = stats.iqr(non_extremal_differences)
        stdev_G_non_extremal = 0.7413 * iqr_non_extremal
        catastrop_outlier_ratio_non_extremal = stdev_differences_non_extremal / stdev_G_non_extremal
        # mean - median test (see Ivezic et al. (2014))
        mean_med_non_extremal = np.mean(non_extremal_differences) - np.median(non_extremal_differences)
        # Calculate Z_1 and Z_2 statistics (see Ivezic et al. (2014))
        Z_1_non_extremal = 1.3 * (np.abs(mean_med_non_extremal)/stdev_differences_non_extremal) * np.sqrt(len(non_extremal_differences))
        Z_2_non_extremal = 1.1 * np.abs(catastrop_outlier_ratio_non_extremal - 1.0) * np.sqrt(len(non_extremal_differences))

    # catastrophic outlier ratio calculation (see Ivezic et al. (2014)):
    stdev_differences = np.std(differences, ddof=1)
    iqr = stats.iqr(differences)
    stdev_G = 0.7413 * iqr
    catastrop_outlier_ratio = stdev_differences / stdev_G
    # mean - median test (see Ivezic et al. (2014))
    mean_med = np.mean(differences) - np.median(differences)
    # Calculate Z_1 and Z_2 statistics (see Ivezic et al. (2014))
    Z_1 = 1.3 * (np.abs(mean_med)/stdev_differences) * np.sqrt(len(differences))
    Z_2 = 1.1 * np.abs(catastrop_outlier_ratio - 1.0) * np.sqrt(len(differences))
    # print the output of different statistical tests for normality
    shap = stats.shapiro(differences)
    if len(non_extremal_indices) > 2:
        shap_non_extremal = stats.shapiro(non_extremal_differences)
    
    if sys.version_info[0] < 3: # if python 2
    
        print >> outfile,"--------------------------------------------------------------------"
        print >> outfile," "
        print >> outfile," "
        print >> outfile,"STATISTICAL TESTS FOR NORMALITY + CHARACTERIZATION OF DISTRIBUTION"
        print >> outfile," "
        print >> outfile," "
        print >> outfile," "
        if len(non_extremal_indices) > 2:
            print >> outfile,"Total sample size: " + str(len(differences)) + "  vs. non-extremal sample size: " + str(len(non_extremal_differences)) + " (" + str((len(non_extremal_differences)/(len(differences)*1.0))*100.) +"%)"
        else:
            print >> outfile,"Total sample size: " + str(len(differences))           
        print >> outfile," "
        print >> outfile," "
        print >> outfile,"Mean - Median test (MM) ~ 0 with sigma ~ 0.76 s /sqrt(N) (s= sample stdev,N = sample size > 100) if normally distributed!"
        print >> outfile," If Gaussian: MM ~ 0 +/- " + str((0.76*stdev_differences)/np.sqrt(len(differences))) + " (95% confidence interval: " + str((0.76*stdev_differences)/np.sqrt(len(differences))*stats.t.ppf((1.95) / 2., len(differences)-1)) + ")"
        print >> outfile,"  Z1 < several sigma"
        print >> outfile,"MM: " + str(mean_med)
        print >> outfile,"Z1: " + str(Z_1)
        if len(non_extremal_indices) > 2:
            print >> outfile," If Gaussian (non-extremal sample): MM ~ 0 +/- " + str((0.76*stdev_differences_non_extremal)/np.sqrt(len(non_extremal_differences))) + " (95% confidence interval: " + str((0.76*stdev_differences_non_extremal)/np.sqrt(len(non_extremal_differences))*stats.t.ppf((1.95) / 2., len(differences)-1)) + ")"
            print >> outfile,"  Z1 < few sigma"
            print >> outfile,"MM (non-extremal sample): " + str(mean_med_non_extremal)
            print >> outfile,"Z1 (non-extremal sample): " + str(Z_1_non_extremal)
        print >> outfile," "
        print >> outfile,"Catastrophic outlier ratio (CAR) ~ 1 with sigma ~ 0.92/sqrt(N) (N = sample size > 100) if normally distributed!"
        print >> outfile," If Gaussian: CAR ~ 1 +/- " + str(0.92/np.sqrt(len(differences))) + " (95% confidence interval: " + str((0.92/np.sqrt(len(differences)))*stats.t.ppf((1.95) / 2., len(differences)-1)) + ")"
        print >> outfile,"  Z2 < few sigma"
        print >> outfile,"CAR:  " + str(catastrop_outlier_ratio)
        print >> outfile,"Z2: " + str(Z_2)
        if len(non_extremal_indices) > 2:
            print >> outfile," If Gaussian: CAR (non-extremal sample) ~ 1 +/- " + str(0.92/np.sqrt(len(non_extremal_differences)))+ " (95% confidence interval: " + str((0.92/np.sqrt(len(non_extremal_differences)))*stats.t.ppf((1.95) / 2., len(non_extremal_differences)-1)) + ")"
            print >> outfile,"  Z2 < several sigma"
            print >> outfile,"CAR (non-extremal sample): " + str(catastrop_outlier_ratio_non_extremal)
            print >> outfile,"Z2 (non-extremal sample): " + str(Z_2_non_extremal)
        print >> outfile," "
        print >> outfile," "
        print >> outfile,"Typical Significance Level Alpha adopted = 0.05"
        print >> outfile," "
        print >> outfile," "
        print >> outfile,"Shapiro-Wilk Results Method " + method + ":"
        print >> outfile,"(p > alpha significance level)"
        print >> outfile,"W-statistic: " + str(shap[0])
        print >> outfile,"p-value: " + str(shap[1])
        if len(non_extremal_indices) > 2:
            print >> outfile,"non-extremal W-statistic: " + str(shap_non_extremal[0])
            print >> outfile,"non-extremal p-value: " + str(shap_non_extremal[1])
        print >> outfile," "    
        ands = stats.anderson(differences,dist='norm')
        if len(non_extremal_indices) > 2:
            ands_non_extremal = stats.anderson(non_extremal_differences,dist='norm')
        print >> outfile,"Anderson-Darling Results Method " + method + ":"
        print >> outfile,"(A2 > Critical value to reject null hypothesis)"
        # If the returned statistic is larger than these critical values then for the corresponding significance level,
        # the null hypothesis that the data come from the chosen distribution can be rejected. 
        print >> outfile,"A2: " + str(ands[0])
        print >> outfile,"Critical:" + str(ands[1])
        print >> outfile,"Significance level:" + str(ands[2])
        if len(non_extremal_indices) > 2:
            print >> outfile,"non-extremal A2: " + str(ands_non_extremal[0])
            print >> outfile,"non-extremal Critical: " + str(ands_non_extremal[1])
            print >> outfile,"non-extremal Significance level: " + str(ands_non_extremal[2])
        print >> outfile," "
        kolm = stats.kstest(differences, 'norm')
        if len(non_extremal_indices) > 2:
            kolm_non_extremal = stats.kstest(non_extremal_differences, 'norm')
        print >> outfile,"Kolmogorov-Smirnov Results Method " + method + ":"
        print >> outfile,"The hypothesis regarding the distributional form is rejected if the test statistic, D, is greater than the critical (p-)value. "
        #This performs a test of the distribution G(x) of an observed random variable against a given distribution F(x). 
        #Under the null hypothesis the two distributions are identical, G(x)=F(x). 
        #The alternative hypothesis can be either ‘two-sided’ (default), ‘less’ or ‘greater’. 
        #The KS test is only valid for continuous distributions.
        print >> outfile,"D: " +  str(kolm[0])
        print >> outfile,"p-value: " + str(kolm[1])
        if len(non_extremal_indices) > 2:
            print >> outfile,"non-extremal D: " + str(kolm_non_extremal[0])
            print >> outfile,"non-extremal p-value: " + str(kolm_non_extremal[1])
        print >> outfile," "
        if len(differences) > 7:
            dagos = stats.normaltest(differences)
            if len(non_extremal_indices) > 7:
                dagos_non_extremal = stats.normaltest(non_extremal_differences)
            print >> outfile,"D'agostino-Pearson Results Method " + method + ":"
            print >> outfile,"(p > alpha significance level)"
            print >> outfile,"K2: " + str(dagos[0])
            print >> outfile,"p-value: " + str(dagos[1])
            if len(non_extremal_indices) > 7:
                print >> outfile,"non-extremal K2: " + str(dagos_non_extremal[0])
                print >> outfile,"non-extremal p-value: " + str(dagos_non_extremal[1])
            else:
                print >> outfile,"No D'agostino-Pearson method possible for (non-extremal) method " + method
            print >> outfile," "
        else:
            print >> outfile,"No D'agostino-Pearson method possible for method " + method
            print >> outfile," "
        print >> outfile,"--------------------------------------------------------------------"
    
    else: # if python 3
        print("--------------------------------------------------------------------")
        print(" ")
        if len(non_extremal_indices) > 2:
            print("Total sample size: " + str(len(differences)) + "  vs. non-extremal sample size: " + str(len(non_extremal_differences)) + " (" + str((len(non_extremal_differences)/(len(differences)*1.0))*100.) +"%)")
        else:
            print("Total sample size: " + str(len(differences)))          
        print(" ")
        print(" ")
        print("Mean - Median test (MM) ~ 0 with sigma ~ 0.76 s /sqrt(N) (s= sample stdev,N = sample size > 100) if normally distributed!")
        print(" If Gaussian: MM ~ 0 +/- " + str((0.76*stdev_differences)/np.sqrt(len(differences))) + " (95% confidence interval: " + str((0.76*stdev_differences)/np.sqrt(len(differences))*stats.t.ppf((1.95) / 2., len(differences)-1)) + ")")
        print("  Z1 < several sigma")
        print("MM: " + str(mean_med))
        print("Z1: " + str(Z_1))
        if len(non_extremal_indices) > 2:
            print(" If Gaussian (non-extremal sample): MM ~ 0 +/- " + str((0.76*stdev_differences_non_extremal)/np.sqrt(len(non_extremal_differences))) + " (95% confidence interval: " + str((0.76*stdev_differences_non_extremal)/np.sqrt(len(non_extremal_differences))*stats.t.ppf((1.95) / 2., len(differences)-1)) + ")")
            print("  Z1 < few sigma")
            print("MM (non-extremal sample): " + str(mean_med_non_extremal))
            print("Z1 (non-extremal sample): " + str(Z_1_non_extremal))
        print(" ")
        print("Catastrophic outlier ratio (CAR) ~ 1 with sigma ~ 0.92/sqrt(N) (N = sample size > 100) if normally distributed!")
        print(" If Gaussian: CAR ~ 1 +/- " + str(0.92/np.sqrt(len(differences))) + " (95% confidence interval: " + str((0.92/np.sqrt(len(differences)))*stats.t.ppf((1.95) / 2., len(differences)-1)) + ")")
        print("  Z2 < few sigma")
        print("CAR:  " + str(catastrop_outlier_ratio))
        print("Z2: " + str(Z_2))
        if len(non_extremal_indices) > 2:
            print(" If Gaussian: CAR (non-extremal sample) ~ 1 +/- " + str(0.92/np.sqrt(len(non_extremal_differences)))+ " (95% confidence interval: " + str((0.92/np.sqrt(len(non_extremal_differences)))*stats.t.ppf((1.95) / 2., len(non_extremal_differences)-1)) + ")")
            print("  Z2 < several sigma")
            print("CAR (non-extremal sample): " + str(catastrop_outlier_ratio_non_extremal))
            print("Z2 (non-extremal sample): " + str(Z_2_non_extremal))
        print(" ")
        print(" ")
        print("Typical Significance Level Alpha adopted = 0.05")
        print(" ")
        print(" ")
        print("Shapiro-Wilk Results Method " + method + ":")
        print("(p > alpha significance level)")
        print("W-statistic: " + str(shap[0]))
        print("p-value: " + str(shap[1]))
        if len(non_extremal_indices) > 2:
            print("non-extremal W-statistic: " + str(shap_non_extremal[0]))
            print("non-extremal p-value: " + str(shap_non_extremal[1]))
        print(" ")      
        ands = stats.anderson(differences,dist='norm')
        if len(non_extremal_indices) > 2:
            ands_non_extremal = stats.anderson(non_extremal_differences,dist='norm')
        print("Anderson-Darling Results Method " + method + ":")
        print("(A2 > Critical value to reject null hypothesis)")
        # If the returned statistic is larger than these critical values then for the corresponding significance level,
        # the null hypothesis that the data come from the chosen distribution can be rejected. 
        print("A2: " + str(ands[0])) 
        print("Critical:" + str(ands[1]))
        print("Significance level:" + str(ands[2]))
        if len(non_extremal_indices) > 2:
            print("non-extremal A2: " + str(ands_non_extremal[0]))
            print("non-extremal Critical: " + str(ands_non_extremal[1]))
            print("non-extremal Significance level: " + str(ands_non_extremal[2]))
        print(" ")
        kolm = stats.kstest(differences, 'norm')
        if len(non_extremal_indices) > 2:
            kolm_non_extremal = stats.kstest(non_extremal_differences, 'norm')
        print("Kolmogorov-Smirnov Results Method " + method + ":")
        print("The hypothesis regarding the distributional form is rejected if the test statistic, D, is greater than the critical (p-)value. ")
        #This performs a test of the distribution G(x) of an observed random variable against a given distribution F(x). 
        #Under the null hypothesis the two distributions are identical, G(x)=F(x). 
        #The alternative hypothesis can be either ‘two-sided’ (default), ‘less’ or ‘greater’. 
        #The KS test is only valid for continuous distributions.
        print("D: " +  str(kolm[0]))
        print("p-value: " + str(kolm[1]))
        if len(non_extremal_indices) > 2:
            print("non-extremal D: " + str(kolm_non_extremal[0]))
            print("non-extremal p-value: " + str(kolm_non_extremal[1]))
        print(" ")
        if len(differences) > 7:
            dagos = stats.normaltest(differences)
            if len(non_extremal_indices) > 7:
                dagos_non_extremal = stats.normaltest(non_extremal_differences)
            print("D'agostino-Pearson Results Method " + method + ":")
            print("(p > alpha significance level)")
            print("K2: " + str(dagos[0]))
            print("p-value: " + str(dagos[1]))
            if len(non_extremal_indices) > 7:
                print("non-extremal K2: " + str(dagos_non_extremal[0]))
                print("non-extremal p-value: " + str(dagos_non_extremal[1]))
            else:
                print("No D'agostino-Pearson method possible for (non-extremal) method " + method)
            print(" ")
        else:
            print("No D'agostino-Pearson method possible for method " + method)
            print(" ")
        print("--------------------------------------------------------------------")
    
    return


def difference_plot(outfile,df_diff,df_e_diff,method1,method2,degreesoffreedom,Critical_t,x_string,percent=False,logplot=False,non_extremal=False):
    plt.figure()
    if non_extremal:
        non_extremal_indices = np.argwhere((df_diff['Difference'] < 1.) & (df_diff['Difference'] > -1.)).flatten() # exclude any differences above 1.5 or below -1.5
        # have to recalculate degrees of freedom
        degreesoffreedom = len(non_extremal_indices)
        p = 0.95
        Critical_t = stats.t.ppf(p, degreesoffreedom-1)
        df_diff = df_diff.copy().iloc[non_extremal_indices] # overwrite a copy of the dataframe in which the extremal values are removed/filtered
        df_e_diff = df_e_diff.copy().iloc[non_extremal_indices] # overwrite copy of error dataframe    
    # calculate percentage differences, as well as log differences and their corresponding propagated uncertainties
    percentdiff = (df_diff['Difference']/df_diff[x_string])*100.
    e_percentdiff = (100./df_diff[x_string]) * np.sqrt(np.power(df_e_diff['Difference'].values,2) + np.power((df_e_diff[x_string]/df_diff[x_string]),2)) # propagated error
    logdiff = np.log10(df_diff['Difference'])
    e_logdiff = 1./(np.log(10.)*df_diff['Difference']) # propagated error
    # plot the differences vs. the reference values (Krouwer) or means (Tukey)
    if percent:
        plt.errorbar(df_diff[x_string],percentdiff,yerr=e_percentdiff,
                 xerr=df_e_diff[x_string],ls='None', color=[ color_dict[u] for u in df_diff['Blazhko/RRLyr'] ])        
    elif logplot:
        plt.errorbar(df_diff[x_string],logdiff,yerr=e_logdiff,
                 xerr=df_e_diff[x_string],ls='None', color=[ color_dict[u] for u in df_diff['Blazhko/RRLyr'] ])        
    else:
        plt.errorbar(df_diff[x_string],df_diff['Difference'],yerr=df_e_diff['Difference'],
                 xerr=df_e_diff[x_string],ls='None', color=[ color_dict[u] for u in df_diff['Blazhko/RRLyr'] ])
    # generate color legend displaying Blazhko/RRlyr difference
    first_legend = plt.legend(handles=patchList, loc=2, frameon=True, fancybox=True, framealpha=1.0)
    first_frame = first_legend.get_frame()
    first_frame.set_facecolor('White')
    ax = plt.gca().add_artist(first_legend)
    # generate plotting space for confidence bounds, limits of agreement, and mean difference
    xspace = np.linspace(df_diff[x_string].values.min()-df_e_diff.iloc[df_diff[x_string].values.argmin()][x_string],
                         df_diff[x_string].values.max()+df_e_diff.iloc[df_diff[x_string].values.argmax()][x_string],
                         num=200)
    # calculate the confidence bounds, limits of agreement and mean difference
    if percent:
        meandiff = percentdiff.mean()
        e_meandiff = percentdiff.std() * np.sqrt(1./degreesoffreedom)        
    elif logplot:
        meandiff = df_diff.mean()['Difference']
        e_meandiff = df_diff.std()['Difference'] * np.sqrt(1./degreesoffreedom)        
    else:
        meandiff = df_diff.mean()['Difference']
        e_meandiff = df_diff.std()['Difference'] * np.sqrt(1./degreesoffreedom)
    conf_meandiff = e_meandiff * Critical_t
    LA_up = np.ones(len(xspace))* (meandiff + 1.96 * e_meandiff * np.sqrt(degreesoffreedom))
    LA_down = np.ones(len(xspace))* (meandiff - 1.96 * e_meandiff * np.sqrt(degreesoffreedom))
    conf_LA = e_meandiff * np.sqrt(3) * Critical_t
    # plot the limits of agreement and mean difference
    plt.plot(xspace,np.ones(len(xspace))* meandiff, color='k',ls='-',label='Mean')
    plt.plot(xspace,LA_down,color='red',ls='--',label='LA')
    plt.plot(xspace,LA_up,color='red',ls='--')
    # generate coloured bands which signify the confidence bounds
    ax = plt.gca()
    ax.fill_between(xspace, meandiff-np.ones(len(xspace))*conf_meandiff, meandiff+np.ones(len(xspace))*conf_meandiff, facecolor='green', alpha=0.2)
    ax.fill_between(xspace, LA_up-np.ones(len(xspace))*conf_LA, LA_up+np.ones(len(xspace))*conf_LA, facecolor='green', alpha=0.2)
    ax.fill_between(xspace, LA_down-np.ones(len(xspace))*conf_LA, LA_down+np.ones(len(xspace))*conf_LA, facecolor='green', alpha=0.2)  
    # generate legend
    Legend = plt.legend(frameon=True, fancybox=True, framealpha=1.0)
    frame = Legend.get_frame()
    frame.set_facecolor('White')
    
    if sys.version_info[0] < 3: # if python 2
    
        # set title and axis labels and print determined limits of agreement
        if x_string=='Reference':
            if non_extremal:
                plt.title('Krouwer Difference plot ' + method1 + " (non-extremal)")
                plt.xlabel('GAIA ' + r'$\varpi$' + ' (mas)')            
                print >> outfile,"----------------------------------------------------------------"
                print >> outfile,'Krouwer Difference plot ' + method1 + " (non-extremal)" + ':'
                print >> outfile," "
                if percent:
                    plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (%)') 
                    print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %"
                    print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %"
                    print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %"
                elif logplot:
                    plt.ylabel('log(GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' +')')    
                    print >> outfile,"log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff)
                    print >> outfile,"log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) 
                    print >> outfile,"log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA) 
                else:
                    plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (mas)')   
                    print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas"
                    print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas" 
                    print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas"
            else:
                plt.title('Krouwer Difference plot ' + method1)
                plt.xlabel('GAIA ' + r'$\varpi$' + ' (mas)')
                print >> outfile,"----------------------------------------------------------------"
                print >> outfile,'Krouwer Difference plot ' + method1 + ':'
                print >> outfile," "
                if percent:
                    plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (%)') 
                    print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %" 
                    print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %"
                    print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %"
                elif logplot:
                    plt.ylabel('log(GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' +')')    
                    print >> outfile,"log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff)
                    print >> outfile,"log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA)  
                    print >> outfile,"log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA)
                else:
                    plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (mas)')   
                    print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas"
                    print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas" 
                    print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas"
            print >> outfile,"----------------------------------------------------------------"
        else:
            print >> outfile,"----------------------------------------------------------------"
            if non_extremal:
                if len(method2) == 0:
                    plt.title('Tukey Mean Difference plot ' + method1 + ' and GAIA (non-extremal)')
                    print >> outfile,'Tukey Mean Difference plot ' + method1 + ' and GAIA' + " (non-extremal)" + ':'
                    print >> outfile," "
                    if percent:
                        plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$'+ ' (%)') 
                        print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %"
                        print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %" 
                        print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %"
                    elif logplot:
                        plt.ylabel('log(GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' +')')    
                        print >> outfile,"log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff)
                        print >> outfile,"log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) 
                        print >> outfile,"log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA) 
                    else:
                        plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (mas)')    
                        print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas"
                        print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas"
                        print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas"
                else:
                    plt.title('Tukey Mean Difference plot ' + method1 + ' and ' + method2 + ' (non-extremal)')
                    print >> outfile,'Tukey Mean Difference plot ' + method1 + ' and ' + method2 + " (non-extremal)" + ':'
                    print >> outfile," "
                    if percent:
                        plt.ylabel(method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$'+ ' (%)')    
                        print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %"
                        print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %"
                        print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %"
                    elif logplot:
                        plt.ylabel('log(' + method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$' +')')    
                        print >> outfile,"log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff) 
                        print >> outfile,"log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA)  
                        print >> outfile,"log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA) 
                    else:
                        plt.ylabel(method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$' + ' (mas)')    
                        print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas"
                        print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas"
                        print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas"            
            else:
                if len(method2) == 0:
                    plt.title('Tukey Mean Difference plot ' + method1 + ' and GAIA')
                    print >> outfile,'Tukey Mean Difference plot ' + method1 + ' and GAIA' + ':'
                    print >> outfile," "
                    if percent:
                        plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$'+ ' (%)') 
                        print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %"
                        print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %"
                        print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %"
                    elif logplot:
                        plt.ylabel('log(GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' +')')    
                        print >> outfile,"log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff) 
                        print >> outfile,"log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) 
                        print >> outfile,"log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA) 
                    else:
                        plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (mas)')    
                        print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas" 
                        print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas"
                        print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas" 
                else:
                    plt.title('Tukey Mean Difference plot ' + method1 + ' and ' + method2)
                    print >> outfile,'Tukey Mean Difference plot ' + method1 + ' and ' + method2 + ':'
                    print >> outfile," "
                    if percent:
                        plt.ylabel(method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$'+ ' (%)')    
                        print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %"
                        print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %"
                        print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %"
                    elif logplot:
                        plt.ylabel('log(' + method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$' +')')    
                        print >> outfile,"log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff)  
                        print >> outfile,"log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) 
                        print >> outfile,"log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA)  
                    else:
                        plt.ylabel(method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$' + ' (mas)')    
                        print >> outfile,"Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas" 
                        print >> outfile,"lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas" 
                        print >> outfile,"upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas"
            plt.xlabel(x_string + ' ' + r'$\varpi$' + ' (mas)')
            print >> outfile,"----------------------------------------------------------------"
    
    else: # if python 3
    
        # set title and axis labels and print determined limits of agreement
        if x_string=='Reference':
            if non_extremal:
                plt.title('Krouwer Difference plot ' + method1 + " (non-extremal)")
                plt.xlabel('GAIA ' + r'$\varpi$' + ' (mas)')            
                print("----------------------------------------------------------------")
                print('Krouwer Difference plot ' + method1 + " (non-extremal)" + ':')
                print(" ")
                if percent:
                    plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (%)') 
                    print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %") 
                    print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %") 
                    print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %") 
                elif logplot:
                    plt.ylabel('log(GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' +')')    
                    print("log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff)) 
                    print("log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) ) 
                    print("log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA)) 
                else:
                    plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (mas)')   
                    print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas") 
                    print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas") 
                    print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas") 
            else:
                plt.title('Krouwer Difference plot ' + method1)
                plt.xlabel('GAIA ' + r'$\varpi$' + ' (mas)')
                print("----------------------------------------------------------------")
                print('Krouwer Difference plot ' + method1 + ':')
                print(" ")
                if percent:
                    plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (%)') 
                    print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %") 
                    print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %") 
                    print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %") 
                elif logplot:
                    plt.ylabel('log(GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' +')')    
                    print("log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff)) 
                    print("log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) ) 
                    print("log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA)) 
                else:
                    plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (mas)')   
                    print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas") 
                    print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas") 
                    print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas") 
            print("----------------------------------------------------------------")
        else:
            print("----------------------------------------------------------------") 
            if non_extremal:
                if len(method2) == 0:
                    plt.title('Tukey Mean Difference plot ' + method1 + ' and GAIA (non-extremal)')
                    print('Tukey Mean Difference plot ' + method1 + ' and GAIA' + " (non-extremal)" + ':')
                    print(" ")
                    if percent:
                        plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$'+ ' (%)') 
                        print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %") 
                        print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %") 
                        print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %") 
                    elif logplot:
                        plt.ylabel('log(GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' +')')    
                        print("log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff)) 
                        print("log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) ) 
                        print("log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA) ) 
                    else:
                        plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (mas)')    
                        print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas") 
                        print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas") 
                        print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas") 
                else:
                    plt.title('Tukey Mean Difference plot ' + method1 + ' and ' + method2 + ' (non-extremal)')
                    print('Tukey Mean Difference plot ' + method1 + ' and ' + method2 + " (non-extremal)" + ':')
                    print(" ")
                    if percent:
                        plt.ylabel(method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$'+ ' (%)')    
                        print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %") 
                        print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %") 
                        print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %") 
                    elif logplot:
                        plt.ylabel('log(' + method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$' +')')    
                        print("log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff) ) 
                        print("log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) ) 
                        print("log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA) ) 
                    else:
                        plt.ylabel(method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$' + ' (mas)')    
                        print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas") 
                        print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas") 
                        print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas")             
            else:
                if len(method2) == 0:
                    plt.title('Tukey Mean Difference plot ' + method1 + ' and GAIA')
                    print('Tukey Mean Difference plot ' + method1 + ' and GAIA' + ':')
                    print(" ")
                    if percent:
                        plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$'+ ' (%)') 
                        print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %") 
                        print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %") 
                        print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %") 
                    elif logplot:
                        plt.ylabel('log(GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' +')')    
                        print("log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff)) 
                        print("log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) ) 
                        print("log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA) ) 
                    else:
                        plt.ylabel('GAIA ' + r'$\varpi$' + ' - PML ' + r'$\varpi$' + ' (mas)')    
                        print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas") 
                        print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas") 
                        print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas") 
                else:
                    plt.title('Tukey Mean Difference plot ' + method1 + ' and ' + method2)
                    print('Tukey Mean Difference plot ' + method1 + ' and ' + method2 + ':')
                    print(" ")
                    if percent:
                        plt.ylabel(method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$'+ ' (%)')    
                        print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " %") 
                        print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " %") 
                        print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " %") 
                    elif logplot:
                        plt.ylabel('log(' + method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$' +')')    
                        print("log(Mean): " + str(meandiff) + " +/- " + str(conf_meandiff) ) 
                        print("log(lower Limit of Agreement): " + str(LA_down[0]) + " +/- " + str(conf_LA) ) 
                        print("log(upper Limit of Agreement): " + str(LA_up[0]) + " +/- " + str(conf_LA) ) 
                    else:
                        plt.ylabel(method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$' + ' (mas)')    
                        print("Mean: " + str(meandiff) + " +/- " + str(conf_meandiff) + " mas") 
                        print("lower Limit of Agreement: " + str(LA_down[0]) + " +/- " + str(conf_LA) + " mas") 
                        print("upper Limit of Agreement: " + str(LA_up[0]) + " +/- " + str(conf_LA) + " mas") 
            plt.xlabel(x_string + ' ' + r'$\varpi$' + ' (mas)')
            print("----------------------------------------------------------------")
    
    return

    
def generate_sorted_df_fit(outfile,df_diff,df_e_diff,fit_res,robust_fit_res,dmatrix,predbound_up,predbound_down,odr_res,x_string,method1,method2=False,nonextremal=False):
    if nonextremal:
        non_extremal_indices = np.argwhere((df_diff['Difference'] < 1.) & (df_diff['Difference'] > -1.)).flatten() # exclude any differences above 1.0 or below -1.0
        df_diff = df_diff.copy().iloc[non_extremal_indices] # overwrite a copy of the dataframe in which the extremal values are removed/filtered
        df_e_diff = df_e_diff.copy().iloc[non_extremal_indices] # overwrite copy of error dataframe    
    # generate the different fit prediction values and put them in a dataframe
    OLS_predict = fit_res.predict(dmatrix)
    RLS_predict = robust_fit_res.predict(dmatrix)
    ODR_predict = odr_res.y
    ODR_predict_x = odr_res.xplus
    predict_df = pd.DataFrame(np.vstack((OLS_predict,RLS_predict,ODR_predict,ODR_predict_x)),index=["OLS","RLS","ODR","ODR_x"],columns=list(predbound_up.index))
    # transpose the dataframe, in order to add the prediction bounds for OLS
    predict_df = predict_df.T
    predict_df['pred_up'] = pd.Series(predbound_up.values, index=predict_df.index)
    predict_df['pred_down'] = pd.Series(predbound_down.values, index=predict_df.index)
    # generate a new dataframe containing the Differences, Reference (GAIA) parallaxes, and Mean parallaxes and their errors!  
    newdf_diff = df_diff.copy()
    newdf_diff['e_Difference'] = pd.Series(df_e_diff['Difference'].values, index=predict_df.index)
    newdf_diff['e_Reference'] = pd.Series(df_e_diff['Reference'].values, index=predict_df.index)
    newdf_diff['e_Mean'] = pd.Series(df_e_diff['Mean'].values, index=predict_df.index)
    # merge the predict dataframe and the difference dataframe
    mergeddf = pd.concat([newdf_diff,predict_df],axis=1)
    # sort the merged dataframe
    sorteddf = mergeddf.sort_values(x_string)
    if x_string == 'Reference':
        whatisfitted = "Krouwer: " + 'GAIA ' + r'$\varpi$' + ' - ' + method1 + ' ' + r'$\varpi$' 
    else:
        if len(method2) != 0:
            whatisfitted = "Tukey: " + method1 + ' ' + r'$\varpi$' + ' - ' + method2 + ' ' + r'$\varpi$' 
        else:
            whatisfitted = "Tukey: " + 'GAIA ' + r'$\varpi$' + ' - ' + method1 + ' ' + r'$\varpi$' 

    if sys.version_info[0] < 3: # if python 2
        # Print the output of the OLS fit:
        print >> outfile,"--------------------------------------------------------------------"
        print >> outfile," "
        print >> outfile," "
        if nonextremal:
            print >> outfile,whatisfitted + " Ordinary Least Squares Regression (OLS) fit results (non-extremal):"
        else:
            print >> outfile,whatisfitted + " Ordinary Least Squares Regression (OLS) fit results:"
        print >> outfile," "
        print >> outfile," "
        print >> outfile,fit_res.summary()
        print >> outfile," "
        print >> outfile," "
        print >> outfile,"--------------------------------------------------------------------"
        
        # Print the output of the RLS fit:
        print >> outfile,"--------------------------------------------------------------------"
        print >> outfile," "
        print >> outfile," "
        if nonextremal:
            print >> outfile,whatisfitted + " Robust Least Squares Regression (RLS) fit results (non-extremal):"
        else:
            print >> outfile,whatisfitted + " Robust Least Squares Regression (RLS) fit results:"
        print >> outfile," "
        print >> outfile," "
        print >> outfile,robust_fit_res.summary()
        print >> outfile," "
        print >> outfile," "
        print >> outfile,"--------------------------------------------------------------------" 
    else: # if python 3
        # Print the output of the OLS fit:
        print("--------------------------------------------------------------------")
        print(" ")
        print(" ")
        if nonextremal:
            print(whatisfitted + " Ordinary Least Squares Regression (OLS) fit results (non-extremal):")
        else:
            print(whatisfitted + " Ordinary Least Squares Regression (OLS) fit results:")
        print(" ")
        print(" ")
        print(fit_res.summary())
        print(" ")
        print(" ")
        print("--------------------------------------------------------------------")
        
        # Print the output of the RLS fit:
        print("--------------------------------------------------------------------")
        print(" ")
        print(" ")
        if nonextremal:
            print(whatisfitted + " Robust Least Squares Regression (RLS) fit results (non-extremal):")
        else:
            print(whatisfitted + " Robust Least Squares Regression (RLS) fit results:")
        print(" ")
        print(" ")
        print(robust_fit_res.summary())
        print(" ")
        print(" ")
        print("--------------------------------------------------------------------")    

    return sorteddf


def fit_plot(outfile,sorteddf,odr_res,sigma_odr,residual_odr,x_string,method,additional_method='',nonextremal=False):
    if len(sorteddf[x_string].values) > 30:
        markersize = 12 # change markersize
    elif len(sorteddf[x_string].values) > 50:
        markersize = 8 # change markersize
    else:
        markersize= 36 # default  
    # set the plotting style
    sns.set_style('darkgrid')
    if nonextremal:
        ODR_fit_plot(outfile,sorteddf[x_string],sorteddf['Difference'],sorteddf['e_'+x_string],
                 sorteddf['e_Difference'],odr_res,sigma_odr,residual_odr,x_string,
                 method,additional_method=additional_method,nonextremal=True)
    else:
        ODR_fit_plot(outfile,sorteddf[x_string],sorteddf['Difference'],sorteddf['e_'+x_string],
                 sorteddf['e_Difference'],odr_res,sigma_odr,residual_odr,x_string,
                 method,additional_method=additional_method,nonextremal=False)        
    plt.figure()
    ax = plt.gca()
    # plot the differences in function of either reference (Krouwer) or mean (Tukey) parallaxes
    plt.scatter(sorteddf[x_string],sorteddf['Difference'], color=[ color_dict[u] for u in sorteddf['Blazhko/RRLyr'] ],s=markersize)
    # add legend containing the distinction between Blazhko/RRLyr
    second_legend = plt.legend(handles=patchList, loc=2,frameon=True, fancybox=True, framealpha=1.0)
    second_frame = second_legend.get_frame()
    second_frame.set_facecolor('White')
    ax.add_artist(second_legend)
    
    # plot the fits of the differences in function of either reference (Krouwer) or mean (Tukey) parallaxes
    plt.plot(sorteddf[x_string],sorteddf['OLS'],label='OLS Fit',c='k')
    plt.plot(sorteddf[x_string],sorteddf['RLS'],label='RLS Fit',c='Yellow')
    plt.plot(sorteddf['ODR_x'],sorteddf['ODR'],label='ODR Fit',c='Green')
    plt.plot(sorteddf[x_string],sorteddf['pred_up'],c='r',ls='--',label='Prediction Bound')
    plt.plot(sorteddf[x_string],sorteddf['pred_down'],c='r',ls='--',label='')
    # generate the correct labels
    if x_string=='Reference':
        if nonextremal:
            plt.title('Fitted ' + r'$\varpi$' + ' Differences ' + '(Krouwer, ' + method +'; non-extremal)')
        else:
            plt.title('Fitted ' + r'$\varpi$' + ' Differences ' + '(Krouwer, ' + method +')')
        plt.xlabel('GAIA ' + r'$\varpi$' + ' (mas)')
        plt.ylabel('GAIA ' + r'$\varpi$' + ' - ' + method + ' ' + r'$\varpi$' + ' (mas)')
    else:
        if len(additional_method) == 0:
            if nonextremal:
                plt.title('Fitted ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', GAIA; non-extremal)')
            else:
                plt.title('Fitted ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', GAIA)')
            plt.ylabel('GAIA ' + r'$\varpi$' + ' - ' + method + ' ' + r'$\varpi$' + ' (mas)')
        else:
            if nonextremal:
                plt.title('Fitted ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', ' + additional_method + '; non-extremal)')
            else:
                plt.title('Fitted ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', ' + additional_method + ')')
            plt.ylabel(method + ' ' + r'$\varpi$' + ' - ' + additional_method + ' ' + r'$\varpi$' + ' (mas)')
        plt.xlabel(x_string + ' ' + r'$\varpi$' + ' (mas)')
    # add the legend containing the distinction between the different fits
    Legend2 = plt.legend(loc=1,frameon=True, fancybox=True, framealpha=1.0)
    frame2= Legend2.get_frame()
    frame2.set_facecolor('White')
    return

def ODR_fit_plot(outfile,x_data,y_data,x_sigma,y_sigma,output,sigma_odr,residual_odr,x_string,method,additional_method='',nonextremal=False):
    # create figure
    fig = plt.figure()
    # Make the subplot
    fit = fig.add_subplot(211)
    # remove tick labels from upper plot
    fit.set_xticklabels( () )
    # set title
    if x_string=='Reference':
        if nonextremal:
            plt.title('Fitted (ODR) ' + r'$\varpi$' + ' Differences ' + '(Krouwer, ' + method +'; non-extremal)')
        else:
            plt.title('Fitted (ODR) ' + r'$\varpi$' + ' Differences ' + '(Krouwer, ' + method +')')
        plt.ylabel('GAIA ' + r'$\varpi$' + ' - ' + method + ' ' + r'$\varpi$' + ' (mas)')
        whatisfitted = "Krouwer: GAIA " + r'$\varpi$' + ' - ' + method + ' ' + r'$\varpi$' 
    else:
        if len(additional_method) == 0:
            if nonextremal:
                plt.title('Fitted (ODR) ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', GAIA; non-extremal)')
            else:
                plt.title('Fitted (ODR) ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', GAIA)')
            plt.ylabel('GAIA ' + r'$\varpi$' + ' - ' + method + ' ' + r'$\varpi$' + ' (mas)')
            whatisfitted = "Tukey: GAIA " + r'$\varpi$' + ' - ' + method + ' ' + r'$\varpi$' 
        else:
            if nonextremal:
                plt.title('Fitted (ODR) ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', ' + additional_method + '; non-extremal)')
            else:
                plt.title('Fitted (ODR) ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', ' + additional_method + ')')
            plt.ylabel(method + ' ' + r'$\varpi$' + ' - ' + additional_method + ' ' + r'$\varpi$' + ' (mas)')
            whatisfitted = "Tukey: " + method + ' ' + r'$\varpi$' + ' - ' + additional_method + ' ' + r'$\varpi$' 
    # Generate linspace for plotting
    stepsize = (max(x_data)-min(x_data))/1000.
    # set plotting margin to 50 times the stepsize (beyond and before max and min, respectively)
    margin = 50*stepsize
    x_model = np.arange( min(x_data)-margin,max(x_data)+margin,
                                    stepsize)    
    # plot the ODR fit
    fit.plot(x_data,y_data,'ro', x_model, f_ODR(output.beta,x_model),markersize=4,label='')
    # Add error bars
    fit.errorbar(x_data, y_data, xerr=x_sigma, yerr=y_sigma, fmt='r+', label='Differences')
    # set y-scale to linear
    fit.set_yscale('linear')
    # draw starting guess (in our case just the x-axis) as dashed green line
    fit.axhline(y=0, c='g',linestyle="-.", label="No Diff")
    
    # output.xplus = x + delta
    a = np.array([output.xplus,x_data])
    # output.y = f(p, xfit), or y + epsilon
    b = np.array([output.y,y_data])
    # plot the actual fit
    fit.plot( a[0][0], b[0][0], 'b-', label= 'Fit')
    # plot the residuals
    fit.plot(np.array([a[0][0],a[1][0]]),np.array([b[0][0],b[1][0]]),'k--', label = 'Residuals')
    for i in range(1,len(y_data)):
        fit.plot( a[0][i], b[0][i], 'b-')
        fit.plot( np.array([a[0][i],a[1][i]]),np.array([b[0][i],b[1][i]]),'k--')
    # plot the legend
    legend = fit.legend(frameon=True, fancybox=True, framealpha=1.0)
    frame= legend.get_frame()
    frame.set_facecolor('White')
    
    # separate plot to show residuals
    residuals = fig.add_subplot(212)
    residuals.errorbar(x=x_data,y=residual_odr,yerr=sigma_odr,fmt='r+',label = "Residuals")
    # make sure residual plot has same x axis as fit plot
    residuals.set_xlim(fit.get_xlim())
    # Draw a horizontal line at zero on residuals plot
    plt.axhline(y=0, color='g')
    # Label axes
    if x_string=='Reference':
        plt.xlabel('GAIA ' + r'$\varpi$' + ' (mas)')
    else:
        plt.xlabel(x_string + ' ' + r'$\varpi$' + ' (mas)')
    # set a plain tick-label style
    plt.ticklabel_format(style='plain', useOffset=False, axis='x')
    plt.ylabel('Residual ' + r'$\varpi$' + ' Difference' + ' (mas)')

    if sys.version_info[0] < 3: # if python 2
   
        # Print the output of the ODR fit:
        print >> outfile,"--------------------------------------------------------------------"
        print >> outfile," "
        print >> outfile," "
        if nonextremal:
            print >> outfile,whatisfitted + " Orthogonal Distance Regression (ODR) fit results (non-extremal):"
        else:
            print >> outfile,whatisfitted + " Orthogonal Distance Regression (ODR) fit results:"
        print >> outfile," "
        print >> outfile," "
        
        # temporarily change stdout to file
        orig_stdout = sys.stdout
        sys.stdout = outfile
        # print the ODR output to file
        output.pprint() 
        # set back to original
        sys.stdout = orig_stdout
        
        print >> outfile," "
        print >> outfile," "
        print >> outfile,"--------------------------------------------------------------------"
    
    else: # if python 3

        # Print the output of the ODR fit:
        print("--------------------------------------------------------------------")
        print(" ")
        print(" ")
        if nonextremal:
            print(whatisfitted + " Orthogonal Distance Regression (ODR) fit results (non-extremal):")
        else:
            print(whatisfitted + " Orthogonal Distance Regression (ODR) fit results:")
        print(" ")
        print(" ")
        output.pprint()
        print(" ")
        print(" ")
        print("--------------------------------------------------------------------")
   
    return

def Bland_Altman_Krouwer_plot(outfile,df_plx,df_diff,df_e_diff,dmatrix,fit_res,predbound_up,predbound_down,robust_fit_res,odr_res,sigma_odr,residual_odr,method1,method2=False,percent=False,logplot=False,nonextremal=False):
    # using the Bland_Altman beta's and alfa's previously calculated, the Bland-Altman plot or the Krouwer plot is generated,
    # as well as returning limits of agreement, with their 95% confidence intervals.

    df_diff['Blazhko/RRLyr'] = pd.Series(df_plx.loc['Blazhko/RRLyr'].values, index=df_diff.index)
    df_e_diff['Blazhko/RRLyr'] = pd.Series(df_plx.loc['Blazhko/RRLyr'].values, index=df_e_diff.index)

    # generate the degrees of freedom, as well as the critical value of the student's t distribution
    degreesoffreedom = df_diff.shape[0]
    p = 0.95
    Critical_t = stats.t.ppf(p, degreesoffreedom-1)

    if  isinstance(method2, basestring):
        if nonextremal:
            # Generate the (non-extremal) Tukey plot
            difference_plot(outfile,df_diff,df_e_diff,method1,method2,degreesoffreedom,Critical_t,'Mean',percent=percent,logplot=logplot,non_extremal=True)        
            # Make a sorted df containing the information needed for a fit plot (non-extremal)
            sorteddf = generate_sorted_df_fit(outfile,df_diff,df_e_diff,fit_res,robust_fit_res,
                                              dmatrix,predbound_up,predbound_down,
                                              odr_res,'Mean',method1,method2=method2,nonextremal=True)
            # Make new figure showing the fits to the differences (non-extremal)
            fit_plot(outfile,sorteddf,odr_res,sigma_odr,residual_odr,'Mean',method1,additional_method=method2,nonextremal=True)             
        else:
            # Generate the Tukey plot
            difference_plot(outfile,df_diff,df_e_diff,method1,method2,degreesoffreedom,Critical_t,'Mean',percent=percent,logplot=logplot)        
            # Make a sorted df containing the information needed for a fit plot
            sorteddf = generate_sorted_df_fit(outfile,df_diff,df_e_diff,fit_res,robust_fit_res,
                                              dmatrix,predbound_up,predbound_down,odr_res,'Mean',method1,method2=method2)
            # Make new figure showing the fits to the differences
            fit_plot(outfile,sorteddf,odr_res,sigma_odr,residual_odr,'Mean',method1,additional_method=method2)  
    else:
        if nonextremal:
            # Generate the (non-extremal) Krouwer plot
            difference_plot(outfile,df_diff,df_e_diff,method1,method2,degreesoffreedom,Critical_t,'Reference',percent=percent,logplot=logplot,non_extremal=True)        
            # Make a sorted df containing the information needed for a fit plot (non-extremal)
            sorteddf = generate_sorted_df_fit(outfile,df_diff,df_e_diff,fit_res,robust_fit_res,
                                              dmatrix,predbound_up,predbound_down,odr_res,'Reference',method1,nonextremal=True)
            # Make new figure showing the fits to the differences (non-extremal)
            fit_plot(outfile,sorteddf,odr_res,sigma_odr,residual_odr,'Reference',method1,nonextremal=True)            
        else:           
            # Generate the Krouwer plot
            difference_plot(outfile,df_diff,df_e_diff,method1,method2,degreesoffreedom,Critical_t,'Reference',percent=percent,logplot=logplot)        
            # Make a sorted df containing the information needed for a fit plot
            sorteddf = generate_sorted_df_fit(outfile,df_diff,df_e_diff,fit_res,robust_fit_res,
                                              dmatrix,predbound_up,predbound_down,odr_res,'Reference',method1)
            # Make new figure showing the fits to the differences
            fit_plot(outfile,sorteddf,odr_res,sigma_odr,residual_odr,'Reference',method1)
    return 
