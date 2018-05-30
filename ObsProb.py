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

X = pd.concat([VT, PEEP, DP, NMBA], axis=1)
collist = list(X.columns)
imp = Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imp.fit(X)
X = imp.transform(X)
X = pd.DataFrame(X,columns=collist)

model = LogisticRegressionCV(cv=5)

scores = cross_val_score(model, X, list(Y['Y']),cv=10)
model_accuracy = round(np.mean(scores),2)
print(round(model_accuracy,2))

print("")
model.fit(X,Y)
col_name = list(X.columns)

model_coef = list(model.coef_[0])
for colname, coef_val in zip(col_name,model_coef):
    print(colname, coef_val)

print("")


N = np.shape(X)[0]
survival_rate = list()

VT_val = 6
NMBA_val = 0

for i in range(N):
    patients_info = X.iloc[[i]]
    patients_idx = X.iloc[[i]].index[0]
    intervened_info = patients_info
    intervened_info = intervened_info.set_value(patients_idx, 'VT_weight', VT_val)
    # intervened_info = intervened_info.set_value(patients_idx, 'NMBA', NMBA_val)
    subject_rate = model.predict_proba(intervened_info)[0][1]
    survival_rate.append(subject_rate)

avg_survival = round(np.mean(survival_rate),2)
print(avg_survival)