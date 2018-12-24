## coding: utf-8 
#!/usr/bin/env python3.7
#
# File: create_GAIA_data_csv.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Generates a text file containing GAIA parallaxes using a crossmatch.csv file and 
# a csv file containing at least the name, period, [Fe/H], RRtype, Vmag, e_Vmag, Kmag, e_Kmag, W1mag and E_W1mag columns

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

from astroquery.vizier import Vizier
import pandas as pd
import re
import numpy as np
from uncertainties import unumpy
import sys

#-------------------------------------------------------
# Allows the user to specify the name under which the file is saved (IF CHANGED, has to be updated in main script!)
savename = "GAIA_DATA_csv.dat"
# Specify path + filenames for all necessary files:
csv_path = '~/GAIA/' # folder in which both csv-files exist
csv_gaia_id = "Xmatch.csv" # cross-match csv
csv_filename = "RRL.csv" # csv containing information mentioned above
#-------------------------------------------------------

Vizier.ROW_LIMIT = -1 # unlimited row limit for query

# load csv with stars and respective useful columns, then drop any rows containing NaNs in them
data_csv = pd.read_csv(csv_path+csv_filename)
data_csv = data_csv[["Name","Period","FeH","RRtype","Vmag","e_Vmag","Kmag","e_Kmag","W1mag","e_W1mag"]]
data_csv = data_csv.dropna(axis=0)

# load in the GAIA crossmatch from another csv file + index using the stars in the previously loaded dataframe
GAIA_X_match = pd.read_csv(csv_path+csv_gaia_id)
GAIA_X_match = GAIA_X_match[GAIA_X_match['Identifier'].isin(data_csv["Name"].tolist())] # only select those identifiers also in the actual dataframe

# find data of all stars contained within the cross match and the data csv, not containing any NaNs
data_csv = data_csv[data_csv["Name"].isin(GAIA_X_match['Identifier'].tolist())]

# all stars to be queried
starnames = GAIA_X_match["Identifier"].tolist()
GAIA_ids = GAIA_X_match['GaiaID'].values.astype(str)

# look up GAIA catalog inputs for the stars
plxs = []
e_plxs = []
starnames_GAIA = []
for starname,GAIA_id in zip(starnames,GAIA_ids):
    star = Vizier.query_object(starname, catalog=["Gaia"])
    for table_name in star.keys():
        dataframe = star[table_name].to_pandas()
        if re.search(r'345\/gaia2',table_name):
            table_of_sourcenames = star[table_name]["Source"]
            dataframe = star[table_name].to_pandas()
            matchindex = [i for i,j in enumerate(table_of_sourcenames) if re.search(GAIA_id,str(j))]
            PLX = dataframe.iloc[matchindex]["Plx"].values
            e_PLX = dataframe.iloc[matchindex]["e_Plx"].values
    plxs.append(PLX[0])
    e_plxs.append(e_PLX[0])
    print(starname)

# convert lists to numpy arrays
PLXs = np.array(plxs).astype(str)
e_PLXs = np.array(e_plxs).astype(str)
starnames = np.array(starnames).astype(str)

# header for text file
headertxt = ["Starname","Parallax","e_Parallax"]
headerstring = '\t'.join(headertxt)

# save the GAIA PARALLAXES to a tab-delimited .dat-file, in order to be read in by the main script
np.savetxt(savename,np.vstack((starnames,PLXs,e_PLXs)).T,delimiter='\t',fmt="%s",header=headerstring)
