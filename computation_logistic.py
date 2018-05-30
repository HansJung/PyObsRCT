from Preparation import *

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.gaussian_process import kernels
import pickle


# Imputation
ARDSNet_2000 = True
ARDSNet_2004 = False
Papazian = False

# np.random.seed(1)
seed_num = 100
np.random.seed(seed_num)

if ARDSNet_2000 or ARDSNet_2004:
    # mask = VT < 10 # Low VT = 1
    # VT = mask * 1
    # X = pd.concat([VT, PP, PEEP, FO2, MV, RR, CRS, DP,
    #                Age, Weight, APACHE, SOFA, SAPS,
    #                Pneumonia, Aspiration, Bacteremia, Septic_shock, Sepsis,
    #                Sex, Berlin, NMBA], axis=1)

    X = pd.concat([VT, DP, PEEP, CRS_old, FO2, Weight, PP,
                   Age, Sex, APACHE_PROB, SOFA, Berlin, Sepsis, Pneumonia,
                   ], axis=1)

if Papazian:
    # mask_VT = VT < 8  # Low VT = 1
    # VT = mask_VT * 1
    #
    # mask_NMBA = NMBA > 0
    # NMBA = mask_NMBA * 1 # 1 means NMBA

    X = pd.concat([VT, DP, PEEP, CRS_old, FO2, Weight, PP,
                   Age, Sex, APACHE_PROB, SOFA, Berlin, Sepsis, Pneumonia, NMBA
                   ], axis=1)

collist = list(X.columns)
imp = Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imp.fit(X)
X = imp.transform(X)
X = pd.DataFrame(X,columns=collist)

n_cv = 10
model = LogisticRegressionCV(cv=n_cv )

scores = cross_val_score(model, X, list(Y['Y']),cv=n_cv )
model_accuracy = round(np.mean(scores),2)
print(round(model_accuracy,2))

print("")
model.fit(X,Y)
col_name = list(X.columns)

model_coef = list(model.coef_[0])
for colname, coef_val in zip(col_name,model_coef):
    print(colname, coef_val)

print("")
print(inclusion_criteria, Mortality, ARDSNet_2000, ARDSNet_2004, Papazian)