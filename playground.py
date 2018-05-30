from scipy.stats import t
import numpy as np

def significant_test(avg1,avg2,std1,std2,n1,n2):
    t_stat = (avg1-avg2) / np.sqrt((std1**2)/n1 + (std2**2)/n2)
    k = min(n1-1,n2-1)
    cdf_val = t.cdf(t_stat,k)
    return cdf_val



LowPEEP = [5, 8, 10, 10, 12, 14, 16, 18]
HighPEEP = [12, 14, 16, 16, 20, 20, 22, 22]
Cutline = [10, 10, 12, 12, 16, 16, 16, 18, 20]
Fo2_Interval = [[0.3, 0.4], [0.4, 0.5], [0.5,0.6],[0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0], [1.0, 1.0]]

num_iter = len(LowPEEP)
num_LowPEEP = 0
num_HighPEEP = 0
ID_LowPEEP = []
ID_HighPEEP = []

for idx in range(num_iter):
    cutline = Cutline[idx]
    Fo2_interval = Fo2_Interval[idx]
    df_HighPEEP = df['SUBJECT_ID'][(df['FiO2'] >= Fo2_interval[0]) &
                                      (df['FiO2'] < Fo2_interval[1]) &
                                      (df['PEEP'] >= cutline) ]

    df_LowPEEP = df['SUBJECT_ID'][(df['FiO2'] >= Fo2_interval[0]) &
                                       (df['FiO2'] < Fo2_interval[1]) &
                                       (df['PEEP'] < cutline)]
    if (Fo2_interval[0] == 1.0):
        df_HighPEEP = df['SUBJECT_ID'][(df['FiO2'] == 1.0) &
                                           (df['PEEP'] >= cutline)]
        df_LowPEEP = df['SUBJECT_ID'][(df['FiO2'] == 1.0) &
                                          (df['PEEP'] < cutline)]
    num_LowPEEP += len(df_LowPEEP)
    num_HighPEEP += len(df_HighPEEP)
    if df_HighPEEP.any():
        ID_HighPEEP.append(list(df_HighPEEP))
    if df_LowPEEP.any():
        ID_LowPEEP.append(list(df_LowPEEP))

ID_LowPEEP_result = []
for sublist in ID_LowPEEP:
    ID_LowPEEP_result += sublist
ID_HighPEEP_result = []
for sublist in ID_HighPEEP:
    ID_HighPEEP_result += sublist
