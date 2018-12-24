# RR-Lyr-Method-Comparison

The    software    package    we    developed for   period-metallicity-luminosity (PML) relation (of RR Lyrae pulsators) validation   purposes   consists   of   several   independently   working    python    scripts    (as    well    as    some    necessary    modules).   It  should  easily  be  extendable to other PML relations/approaches (for example Period-Wesenheit relations). The main script works in python 2.7, whereas the query scripts have been tested/used in 3.6. Future work includes updating the main script to work in 3.6 as well...

# PML validation script:
The 'main_method_comp_script.py' script, referred to as the main PML validation/comparison script in following sections, makes use of several python modules to allow the user to assess the agreement between the GAIA and PML parallaxes. It has several functionalities to gain more insight on the sample you have selected, by allowing the user to print out several characteristics (e.g. the inferred distances to the stars, according to the PML relations). Moreover, the user should specify the options he/she wants, for the analysis of the agreement of the PML relations in the script itself (a parser interface has yet to be developed, although this might not be ideal, given the amount of options provided to the user). 
## PML relation module
The 'PML_relations.py' module contains the definitions of the PML relations whose agreement is tested. When used in the main script, it provides absolute magnitudes, given the required inputs for the PML relation. This should be refined when using other samples.
## PML parallax module
The 'Distance_Parallax.py' module contains the necessary definitions (distance modulus equation) to calculate the distances and parallaxes for the PML relations, taking into account interstellar attenuation. When used in the main script, it provides the PML parallaxes (and can provide explicit distances).
## Tukey's mean difference/Bland-Altman & Krouwer module
The ‘Tukey_Bland_Altman_Krouwer.py’   module   (when   used   in the  main  script)  generates  the  Tukey  mean  difference/Bland-Altman (BA) plots (Altman & Bland 1983; Bland& Altman 1986), as well as the Krouwer plots (Krouwer  2008). In the former the parallax differences are parametrized in function of the mean of the GAIA (DR2) parallax  (Gaia Collaboration et al. 2016, 2018) and the PML parallax. In the latter they are parametrized in function of the GAIA (DR2) parallax. On  top  of  that  the module  generates  different  plots  that  contain different  inferences  of  possible  biases,  by  means  of  several linear regression techniques (as well as their regression diagnostics), and provide one with the results (printed to a file) of the different statistical tests used to verify the assumption that the differences are normally distributed (a necessary assumption when making use of BA/Krouwer plots). For more information the user is deferred to the publications mentioned above.
## Passing-Bablok module
The   ‘Passing_Bablok.py’ module provides our implementation of the Passing-Bablok regression procedure (in python) used  for  PML  validation (Passing & Bablok 2009). When  used  in  the  main script, it generates the different plots needed to analyze agreement, prints the necessary information for the user to a file, and tests the hypotheses β = 1 & α = 0 (i.e. the calculated PML relation parallaxes are similar to GAIA DR2 parallaxes). An important toint to be raised is that the inferred results rely on a (robust) linearity test: the cusum test as defined by Passing & Bablok (2009). In order to make any inferences about the agreement using the β,α parameters, the PML relation needs to pass this cusum test. If both (null) hypotheses are valid (i.e. the inferred parameters within the  confidence interval are equal to 1 & 0), we can conclude the method is in agreement. Otherwise, it is not: deviations for β indicate proportional biases, deviations for α indicate offsets. For more information the user is deferred to the publication mentioned above.

# Query scripts:
Several query scripts based on the Astroquery python package (Ginsburg et al. 2018) were developed in order to easily extract data from the VizieR database (Ochsenbein  et  al.  2000) that will be used in the PML relation validation.
## Blazhko Variability Query
The   create_data_Blahzko_csv.py script reads in a csv file of your liking that should contain the names  of  the  stars  of  which  you  want  to  extract  the  Blazhko variability (according to the GCVS database, Samus et al. 2009). It then constructs a dat file (‘BLAZHKO_csv.dat’) that contains this Blazhko variability as well as the star names, that is read inby the main PML validation/comparison script.
## Apparent  Magnitude & Parallax  Query
The  create_data.py script extracts the apparent magnitudes in the 2MASS Ks passband, as well as the AllWise W1 passband for a specific sample (our sample) of stars, defined by their names in the file itself. Moreover, it simultaneously extracts the GAIA DR2 parallaxes.  It  then  constructs  a  dat  file  (‘W_K_plx.dat’)  that  contains all this information, and is read in by the main PML validation/comparison script.
## Parallax Query
The create_GAIA_data_csv.py script is specifically designed to easily extract GAIA DR2 parallaxes for the Dambis et al. (2013) sample of stars (although it can readily be extended to include query for parallaxes for other datasets), where it reads in two csv-files:  one  containing  the  cross-matched  stars  with  the  GAIA database and the other containing the full sample. It saves the parallaxes in a file called ‘GAIA_DATA_csv.dat’ that contains the parallaxes with corresponding uncertainties as well as the star names, that is read in by the main PML validation/comparison script.

# Interstellar attenuation scripts:
Two different scripts are provided that allow the user to generate the necessary interstellar attenuation information for PML validation.
## Dust table attenuation
The  'Dusttable.py'  script  provides  the user with a means to efficiently query the NASA/IPAC Galactic Dust  Reddening  and  Extinction  tool (https://irsa.ipac.caltech.edu/applications/DUST/),  by  creating  a  file  called ‘sample.csv’  or  ‘sample_CSV.csv’  depending  on  whether  the user wants to obtain information on their manually defined sample (our predefined sample) or the csv-file containing their sample (e.g. the Dambis et  al.  (2013)  sample that we used to test our software package).  This  file  can  then  easily  be  uploaded to  the  tool,  which  subsequently  provides  you  with  the  necessary  attenuation  information,  which  should  be  saved  in  a  file  called ‘Dustmap_output_CSV_table.txt’, in order to be read in by the main script (although this name can easily be changed).
## Monte Carlo attenuation
The dereddening.py script takes photometric data obtained from VizieR in a votable format and calculates the interstellar attenuation/reddening with robust uncertainty estimates based on a monte carlo approach. Care has to be taken when selecting the actual data(set) used, as outliers might be present in the downloaded votable.
