This is a package for computing of predictions for selling data. Note we didn't have time to implement everything dynamically (e.g. filenames).

Inputs
===
Inputs should be stored in a folder 'Datenexporte' which should be stored in the same place like the program itself. For more details you have to read the code of the specific program.
Input-files should be in csv-format and have the following data.

fk_ArtikelBasis_ID_l;Erfasst_am;BestandMin;Bestellt;Beschaffungszeit;Lagerartikel;Inaktiv_b;dominiert_b;Menge;Rechnungs_Nr;FaktDatum;RName;Monat;Jahr;ErstelltAm_d
123;name;23.03.03;;;1;0;;2;;12.01.15;;1;2015;

Outputs
===
The outputs will be created in the folder 'outputs' which should be stored in the same place like the program itself. 

Dependencies
===
The code was developed with a Jupyter Notebook (version 5.4.0). Moreover we use the following libraries.
python 3.6.4
pandas 0.22.0
numpy 1.14.0
matplotlib 2.1.2
statsmodels 0.8.0
match 
itertools
statistics
datetime
scipy 1.0.0
re
os
sys
requests 2.18.4
math
io
sklearn
operator

Directories
===
There is folder 'sarima' in which we placed all code concerning the SARIMA-model. 
The two files in the sarima-folder compute and analyse the results of our main-prediction. 
The comparison with client's prediction is also made in the analysis-file.
In the subfolder 'comparison' there is code for predictions and analysis of comparison we mentioned in our project-report.

In the folder 'grouping' is an own implemented algorithm for grouping the selling data according to the seasonal selling-behavior.

