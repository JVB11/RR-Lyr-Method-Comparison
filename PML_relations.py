## coding: utf-8 
#!/usr/bin/env python2/3
#
# File: PML_relations.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Module that implements different Period-Magnitude-Luminosity relations for RR Lyrae Stars

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

from uncertainties import ufloat
import uncertainties.umath as um  # sin(), etc.
import numpy as np

#------------------------------------------------------------------------------

# Klein et al. 2014: AllWise relations MCMC --> MNRASL 440, L96–L100 (2014)
# Only needs the period in order to calculate the absolute AllWise Magnitudes, 
# in order to compare to the apparent magnitudes from literature, 
# taking into account reddening in order to determine the distance to the star.

def Klein_relations_W1(Period,ePeriod,RRab): # FU/FO W1 MCMC,k
    # define the ufloat
    P = ufloat(Period,ePeriod)
    # first check wether star is a RRab star or whether star is a RRc star!
    if RRab:
        # uncertainties package: a + b*x = y
        a = ufloat(-0.495,0.013)
        b = ufloat(-2.38,0.20)
        Magnitudeuncertainties = a + b*um.log(P/0.55,10)
    else: 
        # uncertainties package: a + b*x = y
        a = ufloat(-0.231,0.031)
        b = ufloat(-1.64,0.62)
        Magnitudeuncertainties = a + b*um.log(P/0.32,10)
    return Magnitudeuncertainties

#------------------------------------------------------------------------------

# Klein et al. 2011: Mv _ k --> The Astrophysical Journal, 738:185 (13pp), 2011 September 10

def Klein_relation_Mv(FeH,eFeH):
    # y = a + b*(Fe + 1.6)
    
    Fe = ufloat(FeH,eFeH)
    a = ufloat(0.56,0.12)
    b = ufloat(0.23,0.04)
    Magnitudeuncertainties = a + b*(Fe + 1.6)
    return Magnitudeuncertainties

#------------------------------------------------------------------------------

# Muraveva et al. 2015: Ks relations MCMC --> The Astrophysical Journal, 807:127 (17pp), 2015 July 10

def Muraveva_relations_LMC(Period,ePeriod,FeH,eFeH,RRab): # FU/FO Ks MCMC LMC
    Fe = ufloat(FeH,eFeH)
    a = ufloat(-1.06,0.01)
    b = ufloat(0.03,0.07)
    c = ufloat(-2.73,0.25)
    if RRab:
        logP = um.log(ufloat(Period,ePeriod),10)
    else:
        logP = um.log(ufloat(Period,ePeriod),10) + 0.127 # fundamentalized
    Magnitudeuncertainties = a + b*Fe + c*logP
    return Magnitudeuncertainties

def Muraveva_relations_plx(Period,ePeriod,FeH,eFeH,RRab): # FU/FO Ks MCMC plx
    Fe = ufloat(FeH,eFeH)
    a = ufloat(-1.25,0.06)
    b = ufloat(0.03,0.07)
    c = ufloat(-2.73,0.25)
    if RRab:
        logP = um.log(ufloat(Period,ePeriod),10)
    else:
        logP = um.log(ufloat(Period,ePeriod),10) + 0.127 # fundamentalized
    Magnitudeuncertainties = a + b*Fe + c*logP
    return Magnitudeuncertainties

#------------------------------------------------------------------------------

# Neeley et al. 2017: W1 relations FO/FU --> The Astrophysical Journal, 841:84 (19pp), 2017 June 1

def Neeley_relations(Period,ePeriod,FeH,eFeH,RRab):
    Fe = ufloat(FeH,eFeH)
    logP = um.log(ufloat(Period,ePeriod),10)
    # Model: y = a + b*Fe + c*log P 
    if RRab: # W1 FU
        a = ufloat(-0.784,0.007)
        b = ufloat(0.183,0.004)
        c = ufloat(-2.274,0.022)
    else: # W1 FO
        a = ufloat(-1.341,0.024)
        b = ufloat(0.152,0.004)
        c = ufloat(-2.716,0.047) 
    Magnitudeuncertainties = a + b*Fe + c*logP
    return Magnitudeuncertainties

#------------------------------------------------------------------------------

# Dambis et al. 2013:  --> MNRAS 435, 3206–3220 (2013)

def Dambis_V_relation(FeH,eFeH): # <M_V>
    # y = a + b*Fe
    Fe = ufloat(FeH,eFeH)
    a = ufloat(1.094,0.091)
    b = ufloat(0.232,0.020)
    Magnitudeuncertainties = a + b*Fe
    return Magnitudeuncertainties

def Dambis_Ks_relation(Period,ePeriod,FeH,eFeH,RRab): # <M_Ks>
    # y = a + b*Fe + c*log(P_f)
    Fe = ufloat(FeH,eFeH)
    a = ufloat(-0.769,0.088)
    b = ufloat(0.088,0.026)
    c = 2.33
    if RRab:
        logP = um.log(ufloat(Period,ePeriod),10)
    else:
        logP = um.log(ufloat(Period,ePeriod),10) + 0.127
    Magnitudeuncertainties = a + b*Fe + c*logP
    return Magnitudeuncertainties

def Dambis_W1_relation(Period,ePeriod,FeH,eFeH,RRab): # <M_W1>
    # y = a + b*Fe + c*log(P_f)
    Fe = ufloat(FeH,eFeH)
    a = ufloat(-0.825,0.088)
    b = ufloat(0.088,0.026)
    c = 2.33
    if RRab:
        logP = um.log(ufloat(Period,ePeriod),10)
    else:
        logP = um.log(ufloat(Period,ePeriod),10) + 0.127
    Magnitudeuncertainties = a + b*Fe + c*logP
    return Magnitudeuncertainties


#------------------------------------------------------------------------------

# Sesar et al. 2017: W1 FU MCMC --> The Astrophysical Journal, 838:107 (8pp), 2017 April 1

# 2 options: 

# 1) perform 'error propagation' using the assymetrical errors,
# requiring me to postulate PDF's in order to gain some knowledge on combined PDF's,
# which can be used to `extract' the assymetric 'propagated' errors.
# For sums, we can use the approach by Roger Barlow 
# --> https://arxiv.org/pdf/physics/0406120.pdf and https://arxiv.org/pdf/physics/0401042.pdf

# 2) symmetrize the assymetrical errors, given that the assymetry is not large --> this approach will be followed

def Sesar_relation(Period,ePeriod,FeH,eFeH,RRab,symmetric):
    if RRab:
        # relation: y = a + b*Fe + c*logP
        stdamin = 0.10
        stdaplus = 0.12
        stdbmin = 0.08
        stdbplus = 0.09
        stdcmin = 0.73
        stdcplus = 0.74
        Fe = ufloat(FeH,eFeH)
        logP = um.log(ufloat(Period,ePeriod),10)  # relation only applicable for RRab !

        sigma_M_max = 0.04
        epsilon_max = ufloat(0,((-2.47*logP.std_dev)**2) + ((-0.42*Fe.std_dev)**2) + ((sigma_M_max)**2))   # only using maximum a posteriori values for sigma M 
        
        if symmetric: # Option 1
            a = ufloat(-2.47,max(stdamin,stdaplus)) # take conservative symmetric error!
            b = ufloat(0.15,max(stdbmin,stdbplus))
            c = ufloat(-0.42,max(stdcmin,stdcplus))
            Magnitudeuncertainties = a + b*(Fe-(-1.4)) + c*(logP-um.log(0.52854,10)) + epsilon_max
        # else: # Option 2
            # create the combined PDF's from which one could potentially extract a more correct error
        return Magnitudeuncertainties
    else:
        print("Warning: The star passed to this relation is not of type RRab. This relation is only suitable for such stars. The end result for this star will be NaN +/- NaN.")
        return ufloat(np.NaN,np.NaN)

#------------------------------------------------------------------------------
