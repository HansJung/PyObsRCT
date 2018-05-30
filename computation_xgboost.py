from Preparation import *

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ARDSNet_2000 = True
ARDSNet_2004 = False
Papazian = False

if ARDSNet_2000 or ARDSNet_2004:
    # mask = VT < 10 # Low VT = 1
    # VT = mask * 1
    X = pd.concat([VT, PEEP, FO2, CRS, DP,
                   Age, Weight, APACHE_PROB, SAPS_PROB, SOFA,
                   Pneumonia, Sepsis, NMBA,
                   Sex, Berlin], axis=1)
if Papazian:
    # mask_VT = VT < 8  # Low VT = 1
    # VT = mask_VT * 1
    #
    # mask_NMBA = NMBA > 0
    # NMBA = mask_NMBA * 1 # 1 means NMBA

    X = pd.concat([VT, PP, PEEP, PIP, MV, RR, FO2,
                   Age, Weight, APACHE,
                   Pneumonia, Aspiration, Bacteremia, Septic_shock, Sepsis,
                   Sex, Berlin, NMBA], axis=1)

X_train, X_test, Y_train, Y_test = \
    train_test_split(X,Y,test_size=0.3,random_state=1)

model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=10000, max_depth=10,
                          reg_lambda=0.5, reg_alpha=0.5, subsample=1, objective='binary:logistic',silent=1)
model.fit(X, Y)
y_pred = model.predict(X)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y, predictions)

feature_importance = model.feature_importances_
feature_importance = pd.DataFrame(feature_importance)
importance_rank = list(feature_importance.rank(ascending=False)[0])
importance_rank = list(map(int,importance_rank))
importance_rank = list(np.array(importance_rank) - 1)
importance_features = list(X.columns)
result_features = [-100] * len(importance_features)
for i in importance_rank:
    result_features[i] = importance_features[importance_rank.index(i)]


print(round(accuracy,2))
#
# filename = 'xgb_170808.sav'
# pickle.dump(model, open(filename, 'wb'))

