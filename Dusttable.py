## coding: utf-8 
#!/usr/bin/env python2/3
#
# File: main_method_comp_script.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Dust Reddening Table creator: script that will create a readable table 
# for use in the NASA/IPAC  INFRARED  SCIENCE  ARCHIVE Galactic Dust Reddening and Exctinction service,
# yielding the E(B-V) values for your sample of stars that can then be read in in the main script

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

import pandas as pd
import numpy as np

###########################################################################################
# Select whether you would like to provide manual input (if True) or provide a file (if False)?
manual = False

# Filepath + filename for non-manual input (reading in csv's)
Filepath = "~/Documents/GAIA/"
filename = "RRL.csv"

# Filename of saved file:
Saved_Filename = "sample"

###########################################################################################


if manual:
    # all J2000 coordinates need to be in decimal degrees!   ----------> from SIMBAD
    star = ['SV Eri','XZ Dra','SW And','RU Psc', 'X Ari','BD+184995','RZ Cep','V1057 Cas']
    ra = [47.9671145875226,287.4275310168180,5.9295407356087,18.6084884379094,
          47.1286850808402,337.0693199945518,339.8049024487712,12.9709236715110]
    dec = [-11.3539074501484,64.8589261858473,29.4010090759625,24.4156582352956,
           10.4458949746998,19.3652911969447,64.8585025696523,65.1805882867786]
    columns = ["id","ra","dec"]
    # create dataframe that will be written to text file
    df = pd.DataFrame(np.vstack((star,ra,dec)).T,columns=columns)
    # write the dataframe to text file
    df.to_csv(Saved_Filename+".csv",index=None)
else:
    # all J2000 coordinates (in CSV) need to be in decimal degrees!
    data_csv = pd.read_csv(Filepath+filename) # load the full csv file in a dataframe
    data_csv = data_csv[["Name","RA","DEC","Period","FeH","RRtype","Vmag","e_Vmag","Kmag","e_Kmag","W1mag","e_W1mag"]] # select a subset of columns
    data_csv = data_csv.dropna(axis=0) # drop columns containing NaNs
    data_csv = data_csv[["Name","RA","DEC"]].rename(index=str, columns={"Name": "id", "RA": "ra", "DEC": "dec"}) # select the useful subset of columns with correct column names
    # write the dataframe to text file
    data_csv.to_csv(Saved_Filename + "_CSV.csv",index=None)

    

