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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)



empresas = ['ecopetrol', 'bancolombia', 'colcap']
clasificadores = ['sentimiento_textblob', 'sentimiento_vader', 'sentimiento_senticnet', 'sentimiento_lm',
                  'sentimiento_aleatorio']
sheets_financial = ['Ecopetrol OHLCV+indicadores', 'Bancolombia OHLCV+indicadores', 'Icolcap OHLCV+indicadores']

time = pd.date_range('2013-01-01', '2019-12-31', freq='D')
df_time = pd.DataFrame(index=time)
dic_financiero = {}
dic_financiero_copy = {}
dic_financiero_titulo = {}
dic_financiero_completo = {}
instancia_estudio = ['_titulo','_completo']
prediction_process_variables = {}


df_stock = pd.read_excel('OHLCV+indicadores.xlsx', sheet_name=sheets_financial[2], index_col=0)
# df_stock = pd.read_csv("dic_financiero_completo['ecopetrol_df'].csv", index_col=0)

df_stock.insert(loc=8, column='retorno', value=0)
df_stock['retorno'] = df_stock['close'] - df_stock['close'].shift()


# df_stock['retorno'] = df_stock['retorno'].shift(1)

df_stock = df_stock[df_stock['retorno'] != 0]
df_stock['retorno_retraso'] = df_stock['retorno'].shift()
df_stock['retorno_retraso_2'] = df_stock['retorno_retraso'].shift()
df_stock['retorno_retraso_3'] = df_stock['retorno_retraso_2'].shift()
df_stock['volume'] = df_stock['volume'] / 1000000000
# dic_financiero[empresas[i] + '_df']['retorno'] = dic_financiero[empresas[i] + '_df']['retorno'].apply(lambda  x: 1 if x >0 else 0)
df_stock.iloc[:, 5:8] = df_stock.iloc[:, 5:8].shift(1)
# df_stock['volume_dif_1'] = df_stock['volume'] - df_stock['volume'].shift(1)
#
# df_stock['hi-lo'] = df_stock['high'] - df_stock['low']
# df_stock['hi-lo_diff_1'] = df_stock['hi-lo'] - df_stock['hi-lo'].shift(1)
df_stock['direccion'] = df_stock['retorno'].apply(lambda x: 1 if x > 0 else 0)
df_stock = df_stock.dropna()
df_stock_test = df_stock.loc['2019-01-03':'2019-12-30']
# df_stock_test = df_stock.loc['2015-01-05':'2016-1-04']
df_stock = df_stock.loc['2013-01-24':'2019-01-02']
# df_stock = df_stock.loc['2013-01-24':'2015-01-02']
X = df_stock.iloc[:, [5,6,7]]
# X = df_stock.iloc[:, [6, 9]]
y = df_stock.iloc[:, -1]
X_test = df_stock_test.iloc[:, [5,6,7]]
# X_test = df_stock_test.iloc[:, [6,9]]
y_test = df_stock_test.iloc[:, -1]

sm_lg = sm.Logit(y,X)
result = sm_lg.fit()
result.summary()


# lg = LogisticRegression(C= 0.01, penalty='l2', solver='liblinear')
# lg = LogisticRegression(C = 0.5455594781168515, penalty= 'l1', solver ='liblinear')
# prediccion = result.predict(X_test)
# param_grid = {"C":np.linspace(0.01, 1, 100), "penalty":['l2'], "solver": ['newton-cg', 'lbfgs', 'liblinear'] }


param_grid = {}
C = [0.00001, 0.0001, 0.001,0.1,1,10,100,1000]
penalty = ['l2']
solver = ['newton-cg', 'lbfgs', 'liblinear']
i = 0
for c in C:
    for p in penalty:
        for s in solver:
            lg_pg = LogisticRegression(C=c, penalty=p, solver=s)
            lg_pg.fit(X,y)
            lg_p = lg_pg.predict(X_test)
            cr = classification_report(y_test,lg_p, output_dict=True)
            i+=1
            param_grid['lg '+str(i)] = [c,p,s,cr['accuracy']]
            print(param_grid['lg '+str(i)])






#
# gs_ls = GridSearchCV(lg, param_grid, cv=[(slice(None), slice(None))])
# gs_ls.fit(X,y)
# #
# gs_ls.best_params_
# gs_ls.best_score_

# lg.fit(X,y)
# lg_p = lg.predict(X_test)
# cr = classification_report(y_test, lg_p)
#




# prediccion_fixed = prediccion.apply(lambda x: 1 if x > treshold else 0)
# report = classification_report(y_test, prediccion_fixed)
# X = {}
# y = {}
# model = {}
# for i in range(len(sheets_financial)):
#     dic_financiero[empresas[i]+ '_df'] = pd.read_excel('OHLCV+indicadores.xlsx', sheet_name=sheets_financial[i], index_col=0)
#
#     dic_financiero[empresas[i] + '_df'].insert(loc=8, column='retorno', value=0)
#     dic_financiero[empresas[i] + '_df']['retorno'] = dic_financiero[empresas[i] + '_df']['close'] - dic_financiero[empresas[i] + '_df']['close'].shift(1)
#
#     dic_financiero[empresas[i] + '_df'] = dic_financiero[empresas[i] + '_df'][dic_financiero[empresas[i] + '_df']['retorno'] != 0]
#     dic_financiero[empresas[i] + '_df']['volume'] = dic_financiero[empresas[i] + '_df']['volume']/1000000000
#     # dic_financiero[empresas[i] + '_df']['retorno'] = dic_financiero[empresas[i] + '_df']['retorno'].apply(lambda  x: 1 if x >0 else 0)
#     for j in range(1,6):
#         dic_financiero[empresas[i] + '_df']['retraso'+str(j)] = dic_financiero[empresas[i] + '_df']['retorno'].shift(j)
#
#
#     dic_financiero[empresas[i] + '_df']['volume_dif_1'] = dic_financiero[empresas[i] + '_df']['volume'] - dic_financiero[empresas[i] + '_df']['volume'].shift(1)
#     dic_financiero[empresas[i] + '_df'] = dic_financiero[empresas[i] + '_df'].dropna()
#     dic_financiero[empresas[i] + '_df']['hi-lo'] = dic_financiero[empresas[i] + '_df']['high'] - dic_financiero[empresas[i] + '_df']['low']
#     dic_financiero[empresas[i] + '_df']['hi-lo_diff_1'] = dic_financiero[empresas[i] + '_df']['hi-lo'] - dic_financiero[empresas[i] + '_df']['hi-lo'].shift(1)
#     dic_financiero[empresas[i] + '_df']['direccion'] = dic_financiero[empresas[i] + '_df']['retorno'].apply(lambda x: 1 if x > 0 else 0)
#     dic_financiero[empresas[i] + '_df'] = dic_financiero[empresas[i] + '_df'].dropna()
#     # dic_financiero[empresas[i] + '_df'] = df_time.merge(dic_financiero[empresas[i] + '_df'],left_index=True, right_index=True, how='left')
#     # dic_financiero_copy[empresas[i] + '_df'] =dic_financiero[empresas[i] + '_df'].copy()
#     prediction_process_variables['X_'+empresas[i]] = dic_financiero[empresas[i] + '_df'].iloc[:,4:16]
#     prediction_process_variables['y_'+empresas[i]] = dic_financiero[empresas[i] + '_df'].direccion
#     # prediction_process_variables['X_train_' + empresas[i]],prediction_process_variables['y_train_' + empresas[i]],prediction_process_variables['X_test_' + empresas[i]],prediction_process_variables['y_train_' + empresas[i]] = train_test_split(prediction_process_variables['X_'+empresas[i]],prediction_process_variables['y_'+empresas[i]], test_size=0.15, random_state=42)
#     prediction_process_variables['model_'+empresas[i]] = LogisticRegression.fit(X = prediction_process_variables['X_'+empresas[i]],y = prediction_process_variables['y_'+empresas[i]])
#     prediction_process_variables['y_pred_' + empresas[i]] = prediction_process_variables['model_'+empresas[i]].predict(prediction_process_variables['X_test_' + empresas[i]])
#     prediction_process_variables['score_'+empresas[i]] = prediction_process_variables['model_' + empresas[i]].score(prediction_process_variables['X_train_' + empresas[i]],prediction_process_variables['y_train_' + empresas[i]])
#
#
#
#
#
#


