from Preparation import *

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import Imputer
from sklearn.gaussian_process import kernels
import pickle


# Imputation
ARDSNet_2000 = True
ARDSNet_2004 = False
Papazian = False

if ARDSNet_2000:
    # X = pd.concat([VT,PP,Age,Berlin,Sex,Weight,PEEP,FO2,NMBA],axis=1)
    # mask = VT < 8
    # VT = mask * 1
    X = pd.concat([VT, PP, PEEP, PIP, MV, RR, FO2, Cstat, DP,
                   Age, Weight, APACHE, APACHE_PROB,
                   Pneumonia, Aspiration, Bacteremia, Septic_shock, Sepsis,
                   Sex, Berlin, NMBA], axis=1)
if ARDSNet_2004:
    X = pd.concat([PEEP, FO2, Age, Berlin, Sex, Weight], axis=1)
if Papazian:
    mask = NMBA > 0
    NMBA = mask * 1
    X = pd.concat([NMBA, Age, Berlin, Sex, Weight], axis=1)

collist = list(X.columns)
imp = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis=0)
imp.fit(X)
X = imp.transform(X)
X = pd.DataFrame(X,columns=collist)
X_train, X_test, Y_train, Y_test = \
    train_test_split(X,Y,test_size=0.1,random_state=1)

# Kernel
# myKernel = kernels.Sum(kernels.Matern(), kernels.RBF())


# myKernel = kernels.Sum(myKernel,kernels.RationalQuadratic())
# myKernel = kernels.Sum(myKernel,kernels.DotProduct())
myKernel = kernels.RBF()
myKernel = kernels.Sum(myKernel, kernels.DotProduct())
myKernel = kernels.Sum(myKernel,kernels.ConstantKernel())
# myKernel = kernels.Product(myKernel, kernels.DotProduct())
# myKernel = kernels.Sum(myKernel,kernels.ConstantKernel())
model = GaussianProcessClassifier(kernel=myKernel,warm_start=True,n_jobs=2)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)


print(round(accuracy,2))
# filename = 'gp.pkl'
# pickle.dump(model, open(filename, 'wb'))