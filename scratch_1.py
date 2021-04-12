import pandas as pd
import statsmodels.discrete.discrete_model as sm
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import random as rd
import unidecode
import re
from textblob import TextBlob
from textblob.tokenizers import SentenceTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from senticnet.senticnet6 import senticnet
from nltk import word_tokenize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle
import operator
import datetime as dt
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SequentialFeatureSelector

# writer = pd.ExcelWriter('OHLCV+indicadores+sentimientos_semanal_mod2.xlsx', engine='xlsxwriter')
# for i in range(3):
#
#     df_semanal = pd.read_excel('OHLCV+indicadores+sentimientos_semanal_mod.xlsx', index_col=0, sheet_name=empresas[i])
#     df_semanal.insert(loc=0, column='day_of_week', value= df_semanal.index.to_series().dt.dayofweek)
#     df_semanal.to_excel(writer, sheet_name=empresas[i])
#
# writer.close()


empresas = ['Ecopetrol', 'Bancolombia', 'Icolcap']
dic_df_stock_final = {}
dic_df_stock_final_scaled_std = {}

for i in range(len(empresas)):

    dic_df_stock_final[empresas[i]] = pd.read_excel('OHLCV+indicadores+sentimientos_semanal_mod.xlsx', sheet_name=empresas[i], index_col=0)
    # dic_df_stock_final[empresas[i]] = dic_df_stock_final[empresas[i]][(dic_df_stock_final[empresas[i]].iloc[:,12] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,13] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,14] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,15] != 0) ]
    # ESTANDARIZACIÓN FEATURES

    scaler_std = StandardScaler()

    dic_df_stock_final_scaled_std[empresas[i]] = dic_df_stock_final[empresas[i]].copy()
    dic_df_stock_final_scaled_std[empresas[i]] = dic_df_stock_final_scaled_std[empresas[i]].drop(columns=['retorno clases', 'retorno_fijo'])

    dic_df_stock_final_scaled_std[empresas[i]]['retorno_lag_2' ] = dic_df_stock_final_scaled_std[empresas[i]]['retorno'].shift(1)
    dic_df_stock_final_scaled_std[empresas[i]] = dic_df_stock_final_scaled_std[empresas[i]].reindex(columns=['days_of_week', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'williams %R', 'mfi', 'macd(12-26)', 'atr(14)', 'adx(14)', 'textblob_1', 'vader_1', 'senticnet_1', 'lm_1', 'textblob_2', 'vader_2', 'senticnet_2', 'lm_2', 'retorno','retorno_lag_2', 'direccion' ])
    dic_df_stock_final_scaled_std[empresas[i]].iloc[:, 1:-1] = scaler_std.fit_transform(dic_df_stock_final_scaled_std[empresas[i]].iloc[:, 1:-1])










### LISTA CON VARIABLES SELECCIONADAS ECOPETROL, BANCOLOMBIA E ICOLCAP PREDICCÓN DIARIA
# [20,21,6,8,19],[20,7,9,10,11,19], [21,7,9,16,19]

### LISTA CON VARIABLES SELECCIONADAS ECOPETROL, BANCOLOMBIA E ICOLCAP PREDICCON SEMANAL
# [20,6,14,19] [20,21,9],[20,21,10,8]


df = dic_df_stock_final_scaled_std[empresas[2]].copy()
df.iloc[:,6:22] = df.iloc[:,6:22].shift(1)
df = df.dropna()

features = [20,21,10,8]

X = df.iloc[:round(df.shape[0] * 0.8),features]
y = df.iloc[:round(df.shape[0] * 0.8), -1]

X_test = df.iloc[round(df.shape[0] * 0.8):, features]
y_test = df.iloc[round(df.shape[0] * 0.8):, -1]

lg_s = LogisticRegression()
lg_s.fit(X,y)
lg_p_s = lg_s.predict(X_test)

dic_coef_inter = dict()
for i in range(len(lg_s.coef_[0])):
    dic_coef_inter[f'{list(df.columns)[features[i]]}'] = round(lg_s.coef_[0][i], 6)
    if i == len(lg_s.coef_[0])-1:
        dic_coef_inter[f'Intersección'] = round(lg_s.intercept_[0], 6)


print(accuracy_score(y_test, lg_p_s), '\n'*2, roc_auc_score(y_test, lg_p_s))

sm_lg_s = sm.Logit(y,X)
result_s = sm_lg_s.fit()
result_s.summary()


from sklearn.linear_model import RidgeCV
ridge = RidgeCV().fit(X, y)
importance_r = np.abs(ridge.coef_)

feature_names = np.array(df.iloc[:,12:20].columns)
for i , j in enumerate(importance_r):
    if importance_r[i] != 0:
        print(f'primer feature {feature_names[i]}: {importance_r[i]}')
        if i > 2:
            break
    else: break
plt.bar(height=importance_r, x=feature_names)
plt.xticks(rotation='vertical')
plt.title(f"Feature importances via coefficients")

plt.rcParams["figure.figsize"] = (6, 6)
plt.show()




from sklearn.linear_model import LassoCV
lasso = LassoCV(max_iter=1000).fit(X, y)
importance = sorted(np.abs(lasso.coef_),reverse=True)
#
feature_names = np.array(df.iloc[:,12:20].columns)
from sklearn.feature_selection import SequentialFeatureSelector


sfs_forward = SequentialFeatureSelector(lasso, n_features_to_select=3,
                                    direction='forward').fit(X, y)

print("Features selected by forward sequential selection: "
  f"{feature_names[sfs_forward.get_support()]}")

sfs_forward = SequentialFeatureSelector(lasso, n_features_to_select=3,
                                    direction='backward').fit(X, y)

print("Features selected by backward sequential selection: "
  f"{feature_names[sfs_forward.get_support()]}")

import warnings
import operator
warnings.filterwarnings("ignore")

param_grid = {}
C = [0.00001, 0.0001, 0.001,0.1,1,10,100,1000]
penalty = ['l2','l1']
solver = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
i = 0
for c in C:
    for p in penalty:
        for s in solver:
            try:
                lg_pg = LogisticRegression(C=c, penalty=p, solver=s)
                lg_pg.fit(X,y)
                lg_p = lg_pg.predict(X_test)
                cr = (roc_auc_score(y_test,lg_p), classification_report(y_test,lg_p, output_dict=True)['accuracy'])
                i+=1
                param_grid[f'C={c}_penalty={p}_solver={s}'] = cr


            except ValueError: continue
print((max(param_grid.items(), key=operator.itemgetter(1))[0],max(param_grid.items(), key=operator.itemgetter(1))[1]))
#
# from sklearn.linear_model import ElasticNetCV
# elasticnet = ElasticNetCV().fit(X, y)
# importance_en = sorted(np.abs(elasticnet.coef_),reverse=True)
#
# feature_names = np.array(df.iloc[:,6:12].columns)
# for i , j in enumerate(importance_en):
#     if importance_en[i] != 0:
#         print(f'primer feature {feature_names[i]}: {importance_en[i]}')
#         if i > 2:
#             break
#     else: break
# plt.bar(height=importance_en, x=feature_names)
# plt.xticks(rotation='vertical')
# plt.title(f"Feature importances via coefficients")
#
# plt.rcParams["figure.figsize"] = (6, 6)
# plt.show()
from sklearn.linear_model import Ridge

# ridge = Ridge().fit(X, y)
# importance_r = sorted(np.abs(ridge.coef_),reverse=True)
#
# feature_names = np.array(df.iloc[:,6:22].columns)
# for i , j in enumerate(importance_r):
#     if importance_r[i] != 0:
#         print(f'primer feature {feature_names[i]}: {importance_r[i]}')
#         if i > 2:
#             break
#     else: break
# plt.bar(height=importance_r, x=feature_names)
# plt.xticks(rotation='vertical')
# plt.title(f"Feature importances via coefficients")
#
# plt.rcParams["figure.figsize"] = (6, 6)
# plt.show()


# from sklearn.feature_selection import SequentialFeatureSelector
#
#
# sfs_forward = SequentialFeatureSelector(elasticnet, n_features_to_select=3,
#                                     direction='forward').fit(X, y)
#
# print("Features selected by forward sequential selection: "
#   f"{feature_names[sfs_forward.get_support()]}")
#
# sfs_forward = SequentialFeatureSelector(elasticnet, n_features_to_select=3,
#                                     direction='backward').fit(X, y)
#
# print("Features selected by backward sequential selection: "
#   f"{feature_names[sfs_forward.get_support()]}")










import warnings
import operator
warnings.filterwarnings("ignore")

param_grid = {}
C = [0.00001, 0.0001, 0.001,0.1,1,10,100,1000]
penalty = ['l2','l1']
solver = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
i = 0
for c in C:
    for p in penalty:
        for s in solver:
            try:
                lg_pg = LogisticRegression(C=c, penalty=p, solver=s)
                lg_pg.fit(X,y)
                lg_p = lg_pg.predict(X_test)
                cr = classification_report(y_test,lg_p, output_dict=True)
                i+=1
                param_grid[f'C={c}_penalty={p}_solver={s}'] = cr['accuracy']


            except ValueError: continue
print(type(max(param_grid.items(), key=operator.itemgetter(1))[1]))
