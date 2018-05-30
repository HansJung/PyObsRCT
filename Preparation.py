'''
Goal
 - Prepare the dataset encodded in csv format
'''

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Berlin score inclusion criteria

# reading data
df = pd.read_csv('Data/Final_v6.csv')
label_encoder_Sex = LabelEncoder()
label_encoder_Y = LabelEncoder()


# column name
#  list(df.columns)
inclusion_criteria = 300
Mortality = '28DAY'
df = df.ix[(pd.to_numeric(df['BERLIN'],errors='coerce') <= inclusion_criteria)]
# df_sepsis = df.ix[(pd.to_numeric(df['Sepsis'],errors='coerce') == 1)]
# df_pneumonia = df.ix[(pd.to_numeric(df['Pneumonia'],errors='coerce') == 1)]
# df = df_pneumonia

SUBJECT_ID = list(df['SUBJECT_ID'])
Y = pd.to_numeric(df[Mortality],errors='coerce')
label_encoder_Y.fit(Y)
Y = label_encoder_Y.transform(Y)
Y = pd.DataFrame(Y,columns=['Y'])

Age = pd.to_numeric(df['Age'],errors='coerce')
Sex = pd.to_numeric(df['Sex_indicator'],errors='coerce')
Weight = pd.to_numeric(df['WEIGHT'],errors='coerce')
Berlin = pd.to_numeric(df['BERLIN'],errors='coerce')
# Berlin = pd.to_numeric(df['Berlin'],errors='coerce')
NMBA = pd.to_numeric(df['NMBA'],errors='coerce')
FO2 = pd.to_numeric(df['FiO2'],errors='coerce')
VT = pd.to_numeric(df['VT_weight'],errors='coerce')
VT_orig = pd.to_numeric(df['VT'],errors='coerce')
PEEP = pd.to_numeric(df['PEEP'],errors='coerce')
PP = pd.to_numeric(df['PP'],errors='coerce')
MAP = pd.to_numeric(df['MAP'],errors='coerce')
MV = pd.to_numeric(df['Minute Volume'],errors='coerce')
DP = pd.to_numeric(df['DP'],errors='coerce')
CRS_new = pd.to_numeric(df['CRS_new'],errors='coerce')
CRS_old = pd.to_numeric(df['CRS_old'],errors='coerce')
RR = pd.to_numeric(df['RR'],errors='coerce')
PIP = pd.to_numeric(df['PIP'],errors='coerce')
HIP = pd.to_numeric(df['HIP'],errors='coerce')
PO2 = pd.to_numeric(df['PaO2'],errors='coerce')
SO2 = pd.to_numeric(df['SpO2'],errors='coerce')
PCO2 = pd.to_numeric(df['PaCO2'],errors='coerce')
pH = pd.to_numeric(df['pH'],errors='coerce')
Pneumonia = pd.to_numeric(df['Pneumonia'],errors='coerce')
Aspiration = pd.to_numeric(df['Aspiration'],errors='coerce')
Bacteremia = pd.to_numeric(df['Bacteremia'],errors='coerce')
Septic_shock = pd.to_numeric(df['Septic_shock'],errors='coerce')
Sepsis = pd.to_numeric(df['Sepsis'],errors='coerce')
APACHE = pd.to_numeric(df['APACHE'],errors='coerce')
APACHE_PROB = pd.to_numeric(df['APACHE_PROB'],errors='coerce')
SAPS = pd.to_numeric(df['SAPS'],errors='coerce')
SAPS_PROB = pd.to_numeric(df['SAPS_PROB'],errors='coerce')
SOFA = pd.to_numeric(df['SOFA'],errors='coerce')

