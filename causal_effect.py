from Preparation import *
# from computation_xgboost import *
# from computation_gp import *
from computation_logistic import *
from scipy.stats import t
import numpy as np

import pickle
from sklearn.metrics import accuracy_score

def significant_test(avg1,avg2,std1,std2,n1,n2):
    t_stat = (avg1-avg2) / np.sqrt((std1**2)/n1 + (std2**2)/n2)
    k = min(n1-1,n2-1)
    cdf_val = t.cdf(t_stat,k)
    return cdf_val

def intervene_PEEP(curr_FO2, curr_PEEP, LowPEEP):
    FO2_PEEP = dict()
    if LowPEEP == True:
        FO2_PEEP[0.3] = [5, 5]
        FO2_PEEP[0.4] = [8, 8]
        FO2_PEEP[0.5] = [8, 10]
        FO2_PEEP[0.6] = [10, 10]
        FO2_PEEP[0.7] = [10, 14]
        FO2_PEEP[0.8] = [14, 14]
        FO2_PEEP[0.9] = [14, 18]
        FO2_PEEP[1.0] = [18, 24]

        min_PEEP = 5
        maxmax_PEEP = 24
        maxmin_PEEP = 18

    else:
        FO2_PEEP[0.3] = [5, 14]
        FO2_PEEP[0.4] = [14, 16]
        FO2_PEEP[0.5] = [16, 20]
        FO2_PEEP[0.6] = [16, 20]
        FO2_PEEP[0.7] = [16, 20]
        FO2_PEEP[0.8] = [20, 22]
        FO2_PEEP[0.9] = [22, 22]
        FO2_PEEP[1.0] = [22, 24]

        min_PEEP = 12
        maxmax_PEEP = 24
        maxmin_PEEP = 22


    if (np.isnan(curr_FO2) == False and curr_FO2 < 0.3):
        intv_FO2 = 0.3
        intv_PEEP = min_PEEP

    elif (np.isnan(curr_FO2) == False and curr_FO2 > 0.9):
        intv_PEEP = maxmax_PEEP

    else:
        intv_PEEP_range = FO2_PEEP[curr_FO2]
        if curr_PEEP < intv_PEEP_range[0]:
            intv_PEEP = intv_PEEP_range[0]
        elif curr_PEEP > intv_PEEP_range[1]:
            intv_PEEP = intv_PEEP_range[1]
        elif curr_PEEP >= intv_PEEP_range[0] and curr_PEEP <= intv_PEEP_range[1]:
            intv_PEEP = curr_PEEP
            # intv_PEEP = PEEP_allowed[np.where(FO2_allowed == curr_FO2)][0]
    return intv_PEEP

N = np.shape(X)[0]
ARDSNet_2000 = True
ARDSNet_2004 = False
Papazian = False


if ARDSNet_2000:
    VT_val = 12 # 6 VS 12
    LowPEEP = True
    DP_list = []

if ARDSNet_2004:
    VT_val = 6
    LowPEEP = False
    DP_list = []


if Papazian:
    VT_val = 6
    intv_NMBA = 150
    LowPEEP = True
    DP_list = []

survival_prob = list()
idx = 0
j_prob_list = []
for i in range(N):
    ''' For each individual'''
    patients_info = X.iloc[[i]]
    patients_idx = X.iloc[[i]].index[0]
    intervened_info = patients_info

    intervened_info = intervened_info.set_value(patients_idx, 'VT_weight', VT_val)

    # DP_list.append(DP_val)

    if Papazian:
        intervened_info = intervened_info.set_value(patients_idx, 'NMBA', intv_NMBA)

    ''' Control confounder variable '''
    sum_adj = 0
    sum_idx = 0
    for j in range(N):
        subject_CRS = CRS_old.iloc[j]
        subject_FO2 = FO2.iloc[j]
        subject_weight = Weight.iloc[j]
        subject_PP = PP.iloc[j]

        subject_PEEP = PEEP.iloc[j]

        intv_PEEP = intervene_PEEP(subject_FO2, subject_PEEP, LowPEEP=LowPEEP)

        # curr_FO2 = round(patients_info.get_value(patients_idx, 'FiO2'), 1)
        # curr_PEEP = patients_info.get_value(patients_idx, 'PEEP')
        # curr_PP = PP.iloc[j]
        # curr_PIP = PIP.iloc[j]

        DP_val = (subject_weight * VT_val) / (subject_CRS)
        intervened_info = intervened_info.set_value(patients_idx, 'DP', DP_val)
        intervened_info = intervened_info.set_value(patients_idx, 'PEEP',intv_PEEP)

        adj_crs = X['CRS_old'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'CRS_old', adj_crs)

        adj_fio2 = X['FiO2'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'FiO2', adj_fio2)

        adj_Weight = X['WEIGHT'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'WEIGHT', adj_Weight)

        adj_PP = X['PP'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'PP', adj_PP)

        adj_Age = X['Age'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'Age', adj_Age)

        adj_Sex = X['Sex_indicator'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'Sex_indicator', adj_Sex)

        adj_APACHE_PROB = X['APACHE_PROB'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'APACHE_PROB', adj_APACHE_PROB)

        adj_SOFA = X['SOFA'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'SOFA', adj_SOFA)

        adj_Berlin = X['BERLIN'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'BERLIN', adj_Berlin)

        adj_Sepsis = X['Sepsis'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'Sepsis', adj_Sepsis)

        adj_Pneumonia = X['Pneumonia'].iloc[j]
        intervened_info = intervened_info.set_value(patients_idx, 'Pneumonia', adj_Pneumonia)

        j_prob = model.predict_proba(intervened_info)[0][1]  # prob of survival
        j_prob_list.append(j_prob)
        sum_adj += j_prob
    sum_adj /= N
    # survival_prob.append(sum_adj)
    survival_prob.append(sum_adj)
    print(i, SUBJECT_ID[i], round((i/N)*100,2) , round(sum_adj,2), sep=',')
    # if i > 100:
    #     break

avg_survival = np.round(np.mean(survival_prob),2)
std_survival = np.round(np.std(survival_prob),2)
avg_jprob = np.round(np.mean(j_prob_list),2)
std_jprob = np.round(np.std(j_prob_list),2)
N_survival = len(survival_prob)

if ARDSNet_2000 or ARDSNet_2004:
    print(ARDSNet_2000, ARDSNet_2004, Papazian, Mortality, inclusion_criteria, seed_num, round(model_accuracy,2), VT_val, '-' ,avg_survival, std_survival, avg_jprob, std_jprob, N_survival)
elif Papazian:
    print(ARDSNet_2000, ARDSNet_2004, Papazian, Mortality, inclusion_criteria, seed_num, round(model_accuracy, 2), VT_val, intv_NMBA, avg_survival,
          std_survival, avg_jprob, std_jprob, N_survival)
