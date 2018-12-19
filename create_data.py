## coding: utf-8 
#!/usr/bin/env python3.7
#
# File: create_data.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Generates the text file for the main dataframe of the main script (Has to be generated because the main script is still written in python 2.7...

import pandas as pd
import numpy as np

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astropy import coordinates
import astropy.units as u

import re

from uncertainties import unumpy

import sys

Vizier.ROW_LIMIT = -1

def obtain_plx_Ks_W1(starname):
    names_table = Simbad.query_objectids(starname)
    for j in np.array(names_table):
        string = j[0].decode("utf-8")
        MASS = re.search(r'2MASS J(\w+)-(\w+)',string)
        MASS2 = re.search(r'2MASS J(\w+)\+(\w+)',string)
        if MASS:
            plus = False
            MASS_1 = MASS.group(1)
            MASS_2 = MASS.group(2)
        elif MASS2:
            plus = True
            MASS_1 = MASS2.group(1)
            MASS_2 = MASS2.group(2)            
        GAIA = re.search(r'Gaia DR2 (\w+)',string)
        if GAIA:
            GAIA_DR2 = GAIA.group(1)    

    star = Vizier.query_object(starname, catalog=["AllWISE","2MASS","Gaia"])

    for table_name in star.keys():
        dataframe = star[table_name].to_pandas()
        if re.search(r'\/allwise',table_name):
            MASS_1_adapt = MASS_1[:-2] + "." + MASS_1[-2]
            MASS_2_adapt = MASS_2[:-1]
            if not plus:
                matchindex = [i for i,j in enumerate(dataframe["AllWISE"]) if re.search(MASS_1_adapt+"\w+\-"+MASS_2_adapt,j.decode("utf-8"))]
                if matchindex:
                    W1Mag = dataframe.iloc[matchindex]["W1mag"].values
                    e_W1Mag = dataframe.iloc[matchindex]["e_W1mag"].values
                else:
                    MASS_1_adapt = MASS_1[:-2] + "."
                    MASS_2_adapt = MASS_2[:-2]
                    matchindex = [i for i,j in enumerate(dataframe["AllWISE"]) if re.search(MASS_1_adapt+"\w+\-"+MASS_2_adapt,j.decode("utf-8"))]
                    if matchindex:
                        W1Mag = dataframe.iloc[matchindex]["W1mag"].values
                        e_W1Mag = dataframe.iloc[matchindex]["e_W1mag"].values
                    else:
                        MASS_1_adapt = MASS_1[:-3] + "\w" + "."
                        MASS_2_adapt = MASS_2[:-3]
                        matchindex = [i for i,j in enumerate(dataframe["AllWISE"]) if re.search(MASS_1_adapt+"\w+\-"+MASS_2_adapt,j.decode("utf-8"))]
                        if matchindex:
                            W1Mag = dataframe.iloc[matchindex]["W1mag"].values
                            e_W1Mag = dataframe.iloc[matchindex]["e_W1mag"].values
                    
            elif plus:
                matchindex = [i for i,j in enumerate(dataframe["AllWISE"]) if re.search(MASS_1_adapt+"\w+\+"+MASS_2_adapt,j.decode("utf-8"))]
                if matchindex:
                    W1Mag = dataframe.iloc[matchindex]["W1mag"].values
                    e_W1Mag = dataframe.iloc[matchindex]["e_W1mag"].values
                else:
                    MASS_1_adapt = MASS_1[:-2] + "."
                    MASS_2_adapt = MASS_2[:-2]
                    matchindex = [i for i,j in enumerate(dataframe["AllWISE"]) if re.search(MASS_1_adapt+"\w+\+"+MASS_2_adapt,j.decode("utf-8"))]
                    if matchindex:
                        W1Mag = dataframe.iloc[matchindex]["W1mag"].values
                        e_W1Mag = dataframe.iloc[matchindex]["e_W1mag"].values
                    else:
                        MASS_1_adapt = MASS_1[:-3] + "\w" + "."
                        MASS_2_adapt = MASS_2[:-3]
                        matchindex = [i for i,j in enumerate(dataframe["AllWISE"]) if re.search(MASS_1_adapt+"\w+\+"+MASS_2_adapt,j.decode("utf-8"))]
                        if matchindex:
                            W1Mag = dataframe.iloc[matchindex]["W1mag"].values
                            e_W1Mag = dataframe.iloc[matchindex]["e_W1mag"].values


        if re.search(r'246\/out',table_name):
            dataframe = star[table_name].to_pandas()
            if not plus:
                matchindex = [i for i,j in enumerate(dataframe["_2MASS"]) if re.search(MASS_1+"\-"+MASS_2,j.decode("utf-8"))]
                if matchindex:
                    Kmag = dataframe.iloc[matchindex]["Kmag"].values
                    e_Kmag = dataframe.iloc[matchindex]["e_Kmag"].values
            elif plus:
                matchindex = [i for i,j in enumerate(dataframe["_2MASS"]) if re.search(MASS_1+"\+"+MASS_2,j.decode("utf-8"))]
                if matchindex:
                    Kmag = dataframe.iloc[matchindex]["Kmag"].values
                    e_Kmag = dataframe.iloc[matchindex]["e_Kmag"].values
        if re.search(r'345\/gaia2',table_name):
            table_of_sourcenames = star[table_name]["Source"]
            dataframe = star[table_name].to_pandas()
            matchindex = [i for i,j in enumerate(table_of_sourcenames) if re.search(GAIA_DR2,str(j))]
            PLX = dataframe.iloc[matchindex]["Plx"].values
            e_PLX = dataframe.iloc[matchindex]["e_Plx"].values
    return W1Mag[0],e_W1Mag[0],Kmag[0],e_Kmag[0],PLX[0],e_PLX[0]



stars = ['SV Eri','XZ Dra','SW And','RU Psc', 'X Ari','BD+184995','RZ Cep','V1057 Cas'] 

W1Mag = []
KMag = []
PLX = []
eW1Mag = []
eKMag = []
ePLX = []

for star in stars:
     W1,e_W1,K,e_K,Pl,e_Pl= obtain_plx_Ks_W1(star)
     W1Mag.append(W1)
     KMag.append(K)
     PLX.append(Pl)
     eW1Mag.append(e_W1)
     eKMag.append(e_K)
     ePLX.append(e_Pl)
     
W1Mag = np.array(W1Mag)
eW1Mag = np.array(eW1Mag)
KMag = np.array(KMag)
eKMag = np.array(eKMag)

uGAIAparallaxes = unumpy.uarray(PLX,ePLX)
#print(uGAIAparallaxes)


# header for text file
headertxt = ["W1Mag","eW1Mag","KMag","eKMag","PLX","ePLX"]
headerstring = '\t'.join(headertxt)

# save text file containing the W1, Ks magnitudes and GAIA parallaxes
np.savetxt('W_K_plx.dat',np.vstack((W1Mag,eW1Mag,KMag,eKMag,PLX,ePLX)).T,delimiter='\t',header=headerstring)


