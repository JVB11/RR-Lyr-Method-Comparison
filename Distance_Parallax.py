## coding: utf-8 
#!/usr/bin/env python2/3
#
# File: Distance_Parallax.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Module that implements the distance/parallax estimation for different Period-Magnitude-Luminosity relations for RR Lyrae Stars

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
from uncertainties import ufloat
from uncertainties import unumpy
import pandas as pd

#------------------------------------------------------------------------------
# Define an exception class for when we don't have a match
class NotAMatchError(Exception):
    pass

def match_err(string):
    raise NotAMatchError(string)
#------------------------------------------------------------------------------

def get_extinctions(ext_df,R_error,our_script):
    # E(B-V) values
    EBV_S_F = unumpy.uarray(ext_df.loc['EBV_S_F'].values,ext_df.loc['e_EBV_S_F'].values)
    EBV_SFD = unumpy.uarray(ext_df.loc['EBV_SFD'].values,ext_df.loc['e_EBV_SFD'].values)
    # V extinctions
    R_V = np.ones_like(ext_df.loc['EBV_S_F'].values)*3.1
    if R_error:
        e_R_V = np.ones_like(ext_df.loc['EBV_S_F'].values)*0.1
        ext_S_F_V = unumpy.uarray(R_V,e_R_V)*EBV_S_F
        ext_SFD_V = unumpy.uarray(R_V,e_R_V)*EBV_SFD
    else:
        ext_S_F_V = unumpy.uarray(R_V,np.zeros_like(R_V))*EBV_S_F
        ext_SFD_V = unumpy.uarray(R_V,np.zeros_like(R_V))*EBV_SFD
        
    # Ks extinctions
    R_S_F_Ks = np.ones_like(ext_df.loc['EBV_S_F'].values)*0.310
    R_SFD_Ks = np.ones_like(ext_df.loc['EBV_S_F'].values)*0.382
    if R_error:
        e_R_S_F_Ks = np.ones_like(ext_df.loc['EBV_S_F'].values)*0.001
        e_R_SFD_Ks = np.ones_like(ext_df.loc['EBV_S_F'].values)*0.001
        ext_S_F_Ks = unumpy.uarray(R_S_F_Ks,e_R_S_F_Ks)*EBV_S_F
        ext_SFD_Ks = unumpy.uarray(R_SFD_Ks,e_R_SFD_Ks)*EBV_SFD
    else:
        ext_S_F_Ks = unumpy.uarray(R_S_F_Ks,np.zeros_like(R_S_F_Ks))*EBV_S_F
        ext_SFD_Ks = unumpy.uarray(R_SFD_Ks,np.zeros_like(R_SFD_Ks))*EBV_SFD
    
    # W1 values
    R_S_F_W1 = np.ones_like(ext_df.loc['EBV_S_F'].values)*0.189
    R_SFD_W1 = np.ones_like(ext_df.loc['EBV_S_F'].values)*0.234
    if R_error:
        e_R_S_F_W1 = np.ones_like(ext_df.loc['EBV_S_F'].values)*0.001
        e_R_SFD_W1 = np.ones_like(ext_df.loc['EBV_S_F'].values)*0.001
        ext_S_F_W1 = unumpy.uarray(R_S_F_W1,e_R_S_F_W1)*EBV_S_F
        ext_SFD_W1 = unumpy.uarray(R_SFD_W1,e_R_SFD_W1)*EBV_SFD
    else:
        ext_S_F_W1 = unumpy.uarray(R_S_F_W1,np.zeros_like(R_S_F_W1))*EBV_S_F
        ext_SFD_W1 = unumpy.uarray(R_SFD_W1,np.zeros_like(R_SFD_W1))*EBV_SFD
    # if we also use our dereddening script
    if our_script:
        ext_V = unumpy.uarray(ext_df.loc['V'].values,ext_df.loc['e_V'].values)
        ext_Ks = unumpy.uarray(ext_df.loc['Ks'].values,ext_df.loc['e_Ks'].values)
        ext_W1 = unumpy.uarray(ext_df.loc['W1'].values,ext_df.loc['e_W1'].values)
        # Create list of names of rows of df
        rows_df = ['ext_V','ext_V_S_F','ext_V_SFD','ext_Ks','ext_Ks_S_F',
                   'ext_Ks_SFD','ext_W1','ext_W1_S_F','ext_W1_SFD','RRABC']
        # Return df containing the extinction factors for different passbands
        return pd.DataFrame(np.vstack((ext_V,ext_S_F_V,ext_SFD_V,ext_Ks,ext_S_F_Ks,
                                       ext_SFD_Ks,ext_W1,ext_S_F_W1,ext_SFD_W1,
                                       ext_df.loc['RRAB/RRC'].values)),index=rows_df, columns=list(ext_df))
    else:
        # Create list of names of rows of df
        rows_df = ['ext_V_S_F','ext_V_SFD','ext_Ks_S_F',
                   'ext_Ks_SFD','ext_W1_S_F','ext_W1_SFD','RRABC']
        # Return df containing the extinction factors for different passbands
        return pd.DataFrame(np.vstack((ext_S_F_V,ext_SFD_V,ext_S_F_Ks,
                                       ext_SFD_Ks,ext_S_F_W1,ext_SFD_W1,
                                       ext_df.loc['RRAB/RRC'].values)),index=rows_df, columns=list(ext_df))

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------

relationtopassband = {"Dambis Ks": "Ks", "Dambis V": "V","Dambis W1": "W1",
                      "Klein Mv": "V", "Klein W1": "W1", "Muraveva LMC": "Ks",
                      "Muraveva plx": "Ks", "Neeley ": "W1", "Sesar ": "W1"}

def calc_dist(m,M,A):
    # definition that calculates the distance to the (RR Lyr) stars according to the distance modulus equation
    return np.power(10.,0.2*(m-M+5-A))

def row_names_dists(ext_names_list,relation): # utility function that automatically generates names for the different PML, combined with different dereddening procedures
    dict_names = {2:"",3:"_S_F",4:"_SFD"}
    if len(relation.split()) == 1:
        return [relation.split()[0]+dict_names[len(j.split('_'))]for j in ext_names_list]
    else:
        return [relation.split()[0]+dict_names[len(j.split('_'))]+" "+relation.split()[1] for j in ext_names_list]    


def Distance_modulus_distances(M_uncert_df,m_uncert_df,ext_uncert_df,first_relation,second_relation):
    PMLs = list(M_uncert_df.index)
    if first_relation in PMLs: 
        M1 = M_uncert_df.loc[first_relation].values
    else: 
        print("Error: cannot find PML relation.") 
        return False
    if isinstance(second_relation, basestring):
        if second_relation in PMLs: 
            M2 = M_uncert_df.loc[second_relation].values
        else: 
            print("Error, cannot find second specified PML relation.") 
            return False
        m1 = m_uncert_df.loc[relationtopassband[first_relation]].values
        m2 = m_uncert_df.loc[relationtopassband[second_relation]].values
        indices_1 = [i for i, s in enumerate(list(ext_uncert_df.index)) if relationtopassband[first_relation] in s]
        indices_2 = [j for j, k in enumerate(list(ext_uncert_df.index)) if relationtopassband[second_relation] in k]
        first_distances = [calc_dist(m1,M1,ext_uncert_df.iloc[indices_1].iloc[o].values) for o in range(len(ext_uncert_df.iloc[indices_1]))]
        first_row_names = row_names_dists(list(ext_uncert_df.iloc[indices_1].index),first_relation)
        second_distances = [calc_dist(m2,M2,ext_uncert_df.iloc[indices_2].iloc[o].values) for o in range(len(ext_uncert_df.iloc[indices_2]))]
        second_row_names = row_names_dists(list(ext_uncert_df.iloc[indices_2].index),second_relation)
        return pd.DataFrame(first_distances,index=first_row_names,columns=list(ext_uncert_df)),pd.DataFrame(second_distances,index=second_row_names,columns=list(ext_uncert_df))
    else:
        m1 = m_uncert_df.loc[relationtopassband[first_relation]].values
        indices = [i for i, s in enumerate(list(ext_uncert_df.index)) if relationtopassband[first_relation] in s]
        distances = [calc_dist(m1,M1,ext_uncert_df.iloc[indices].iloc[o].values) for o in range(len(ext_uncert_df.iloc[indices]))]
        row_names = row_names_dists(list(ext_uncert_df.iloc[indices].index),first_relation)
        return pd.DataFrame(distances,index=row_names,columns=list(ext_uncert_df))
        
def Distance_to_PLX(df_distances,RRABC=False,BRR=False,second_relation=False,df_distances2 = False,GAIA=False):

    if (isinstance(RRABC,bool) & isinstance(BRR,bool)):
        plx = (1./df_distances)*1000.
        # calculate parallax in mas (based on distances in pc) 
        return plx 
    else:
        if second_relation:
            plx1 = (1./df_distances)*1000.
            plx2 = (1./df_distances2)*1000.
            new_row_names = list(pd.concat([plx1,plx2]).index)
            new_row_names.extend(["RRAB/RRC","Blazhko/RRLyr"])        
            # calculate parallax in mas (based on distances in pc) 
            return pd.DataFrame(np.vstack((plx1,plx2,RRABC,BRR)),index=new_row_names,columns=list(plx1))
        else:
            plx = (1./df_distances)*1000.
            new_row_names = list(plx.index)
            new_row_names.extend(["GAIA","RRAB/RRC","Blazhko/RRLyr"])
            # calculate parallax in mas (based on distances in pc) 
            return pd.DataFrame(np.vstack((plx,GAIA,RRABC,BRR)),index=new_row_names,columns=list(plx)) 
