from scipy.stats import t
import numpy as np
from Preparation import *

def significant_test(avg1,avg2,std1,std2,n1,n2):
    t_stat = (avg1-avg2) / np.sqrt((std1**2)/n1 + (std2**2)/n2)
    k = min(n1-1,n2-1)
    cdf_val = t.cdf(t_stat,k)
    return cdf_val

def conduct_test(df1, df2, name_val, round_idx):
    df_group1 = pd.to_numeric(df1[name_val],errors='coerce')
    df_group2 = pd.to_numeric(df2[name_val],errors='coerce')

    mean1 = round(np.mean(df_group1),round_idx)
    mean2 = round(np.mean(df_group2),round_idx)
    std1 = round(np.std(df_group1),round_idx)
    std2 = round(np.std(df_group2),round_idx)
    N1 = len(df_group1)
    N2 = len(df_group2)
    sigval = significant_test(mean1,mean2,std1,std2,N1,N2)
    if sigval > 0.5:
        sigval = 1-sigval
    sigtype = 0
    if sigval >= 0.01:
        sigtype = 'Insignificant'
    elif sigval <0.01 and sigval > 0.001:
        sigtype = 'sig1'
    elif sigval < 0.001:
        sigtype = 'sig2'

    return [mean1, std1, mean2, std2, sigval,N1,N2, sigtype]

def count_test(df1,df2,name_val):
    df_group1 = pd.to_numeric(df1[name_val], errors='coerce')
    df_group2 = pd.to_numeric(df2[name_val], errors='coerce')

    N1 = len(df_group1)
    N2 = len(df_group2)

    S1 = sum(df_group1)
    S2 = sum(df_group2)

    return [S1,S2, round(S1/N1,2), round(S2/N2,2),N1,N2]

mask_low = df['VT_weight'] <= 8
mask_high = df['VT_weight'] > 8
df_low = df.loc[mask_low]
df_high = df.loc[mask_high]

Age_test = conduct_test(df_low,df_high,'Age',0)
APACHE_test = conduct_test(df_low,df_high,'APACHE',0)
SAPS_test = conduct_test(df_low,df_high,'SAPS',0)
SOFA_test = conduct_test(df_low,df_high,'SOFA',1)
Berlin_test = conduct_test(df_low,df_high,'BERLIN',0)
VT_test = conduct_test(df_low,df_high,'VT',0)
VTweight_test = conduct_test(df_low,df_high,'VT_weight',1)
MV_test = conduct_test(df_low,df_high,'Minute Volume',1)
PEEP_test = conduct_test(df_low,df_high,'PEEP',1)
PP_test = conduct_test(df_low,df_high,'PP',0)
PIP_test = conduct_test(df_low,df_high,'PIP',0)
MAP_test = conduct_test(df_low,df_high,'MAP',0)
RR_test = conduct_test(df_low,df_high,'RR',0)
FO2_test = conduct_test(df_low,df_high,'FiO2',2)
PO2_test = conduct_test(df_low,df_high,'PaO2',0)
PCO2_test = conduct_test(df_low,df_high,'PaCO2',0)
pH_test = conduct_test(df_low,df_high,'pH',2)
DP_test = conduct_test(df_low,df_high,'DP_CRS',0)


Pneumonia_test = count_test(df_low,df_high,'Pneumonia')
Sepsis_test = count_test(df_low,df_high,'Sepsis')
Aspiration = count_test(df_low,df_high,'Aspiration')
Septic_test = count_test(df_low,df_high,'Septic_shock')
Bacteremia_test = count_test(df_low,df_high,'Bacteremia')