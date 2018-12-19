# coding: utf-8 
#!/usr/bin/env python3.7
#
# File: create_GAIA_data_csv.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Generates the text file for the main dataframe of the main script (Has to be generated because the main script is still written in python 2.7...

from astroquery.vizier import Vizier
import pandas as pd
import re
import numpy as np

Vizier.ROW_LIMIT = -1 # unlimited row limit for query

csv_path = '/Users/jvb/Documents/Proposal2018/GAIA/'
csv_gaia_id = "DambisGaiaXmatch.csv"
csv_filename = "RRL_Dambis2013.csv"

# load csv with stars and respective useful columns, then drop any rows containing NaNs in them
data_csv = pd.read_csv(csv_path+csv_filename)
data_csv = data_csv[["Name","Period","FeH","RRtype","Vmag","e_Vmag","Kmag","e_Kmag","W1mag","e_W1mag"]]
data_csv = data_csv.dropna(axis=0)
# get all the starnames
starnames = data_csv["Name"].tolist()



blazhkos = []
starnames_blazhkos = []
for starname in starnames:
    print(starname)
    star = Vizier.query_object(starname, catalog=["AAVSO"])
    for table_name in star.keys():
        if re.search(r'B\/vsx\/vsx',table_name):
            dataframe = star[table_name].to_pandas()
            RRtype = dataframe["Type"].values[0].decode('utf-8')
            if "RRAB" in RRtype: 
                if re.search('RRAB\/\w+',RRtype):
                    blazhkos.append("Blazhko")
                    starnames_blazhkos.append(starname)
                else:
                    blazhkos.append("RRLyr")
                    starnames_blazhkos.append(starname)
            if "RRC" in RRtype:
                if re.search('RRC\/\w+',RRtype):
                    blazhkos.append("Blazhko")
                    starnames_blazhkos.append(starname)
                else:
                    blazhkos.append("RRLyr")
                    starnames_blazhkos.append(starname)
# convert to numpy arrays of string dtype
blazhkos = np.array(blazhkos).astype(str)
starnames_blazhkos = np.array(starnames_blazhkos).astype(str)

# header for text file
headertxt = ["Starname","Blazhko"]
headerstring = '\t'.join(headertxt)

# save the GAIA PARALLAXES to a tab-delimited .dat-file, in order to be read in by the main script
np.savetxt('BLAZHKO_csv.dat',np.vstack((starnames_blazhkos,blazhkos)).T,delimiter='\t',fmt="%s",header=headerstring)