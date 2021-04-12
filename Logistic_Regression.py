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
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import datetime as dt
sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)

with open('pickle_sentimientos_mod.pkl', 'rb') as file:
    dic_stock_sentiments = pickle.load(file)

##### CONCATENACIÓN DATAFRAMES SENTIMIENTOS Y FINANCIEROS


empresas = ['Ecopetrol', 'Bancolombia', 'Icolcap']
clasificadores = ['sentimiento_textblob', 'sentimiento_vader', 'sentimiento_senticnet', 'sentimiento_lm',
                  'sentimiento_aleatorio']
sheets_financial = ['Ecopetrol OHLCV+indicadores', 'Bancolombia OHLCV+indicadores', 'Icolcap OHLCV+indicadores']


time = pd.date_range('2013-01-01', '2019-12-31', freq='D')

df_time = pd.DataFrame(index=time)


instancia_estudio = ['_completo','_titulo'] # los clasificadores para cada instancia tienen una x (completo) y una y (titulo) al final del string

dic_df_stock_final = {}
dic_df_stock_final_copy ={}
dic_df_stock_final_copy2 ={}

for i in range(len(empresas)):

    dic_df_stock_final[empresas[i]] = df_time.merge(pd.read_excel('OHLCV+indicadores.xlsx', index_col=0, sheet_name=f'{empresas[i]} OHLCV+indicadores'), right_index=True, left_index=True, how='left')
    dic_df_stock_final[empresas[i]] = dic_df_stock_final[empresas[i]].merge(dic_stock_sentiments[f'df_{empresas[i].lower()}_completo'], right_index=True, left_index=True, how='left')
    dic_df_stock_final[empresas[i]] = dic_df_stock_final[empresas[i]].merge(dic_stock_sentiments[f'df_{empresas[i].lower()}_titulo'], right_index=True, left_index=True, how='left')
    dic_df_stock_final[empresas[i]].insert(loc=0, column='days_of_week', value= dic_df_stock_final[empresas[i]].index.to_series().dt.dayofweek)
    dic_df_stock_final_copy[empresas[i]] = dic_df_stock_final[empresas[i]].copy()
    dic_df_stock_final_copy2[empresas[i]] = dic_df_stock_final[empresas[i]][(dic_df_stock_final[empresas[i]].iloc[:,0]== 0)| (dic_df_stock_final[empresas[i]].iloc[:,0] == 5) |(dic_df_stock_final[empresas[i]].iloc[:,0] == 6)]
    dic_df_stock_final_copy2[empresas[i]] = dic_df_stock_final_copy2[empresas[i]].iloc[:, 14:22].fillna(0)
    j = 0
    lista_sentimiento_lunes = []
    dic_df_stock_final_copy[empresas[i]] = dic_df_stock_final_copy[empresas[i]].reset_index()
    dic_df_stock_final_copy[empresas[i]] = dic_df_stock_final_copy[empresas[i]].set_index('days_of_week')
    for idx in range(0, 1095, 3):
        if dic_df_stock_final_copy2[empresas[i]].iloc[idx, :].sum() == 0:
            j += 1
        if dic_df_stock_final_copy2[empresas[i]].iloc[idx + 1, :].sum() == 0:
            j += 1
        if dic_df_stock_final_copy2[empresas[i]].iloc[idx + 2, :].sum() == 0:
            j += 1
        if j == 3:
            lista_sentimiento_lunes.append(dic_df_stock_final_copy2[empresas[i]].iloc[idx, :] + dic_df_stock_final_copy2[empresas[i]].iloc[idx + 1, :] + dic_df_stock_final_copy2[empresas[i]].iloc[idx + 2, :])
            j = 0
            continue
        lista_sentimiento_lunes.append((dic_df_stock_final_copy2[empresas[i]].iloc[idx, :] + dic_df_stock_final_copy2[empresas[i]].iloc[idx + 1, :] + dic_df_stock_final_copy2[empresas[i]].iloc[idx + 2, :]) / (3 - j))
        j = 0
    k = 0
    fila = list(range(6,2556, 7))

    for index, row in dic_df_stock_final_copy[empresas[i]].iterrows():
        if index == 0:
            dic_df_stock_final_copy[empresas[i]].iloc[fila[k], 14:22] = lista_sentimiento_lunes[k].values
            k += 1
        else:
            continue

    dic_df_stock_final_copy[empresas[i]].iloc[:, 14:22] = dic_df_stock_final_copy[empresas[i]].iloc[:, 14:22].fillna(0)
    dic_df_stock_final_copy[empresas[i]] = dic_df_stock_final_copy[empresas[i]].dropna()
    dic_df_stock_final_copy[empresas[i]] = dic_df_stock_final_copy[empresas[i]].reset_index()
    dic_df_stock_final_copy[empresas[i]] = dic_df_stock_final_copy[empresas[i]].set_index('index')
    dic_df_stock_final_copy[empresas[i]] = dic_df_stock_final_copy[empresas[i]][dic_df_stock_final_copy[empresas[i]]['retorno'] != 0]
    dic_df_stock_final[empresas[i]] = dic_df_stock_final_copy[empresas[i]].copy()
    dic_df_stock_final[empresas[i]]['retorno_fijo'] = dic_df_stock_final[empresas[i]]['retorno'].copy()
    dic_df_stock_final[empresas[i]]['direccion'] = dic_df_stock_final[empresas[i]]['retorno'].apply(lambda x: 1 if x>0 else 0)
    dic_df_stock_final[empresas[i]] = dic_df_stock_final[empresas[i]].reindex(columns=['days_of_week', 'open', 'high', 'low', 'close', 'volume', 'rsi',
       'williams %R', 'mfi', 'macd(12-26)',
       'atr(14)','adx(14)', 'sentimiento_textblob_x',
       'sentimiento_vader_x', 'sentimiento_senticnet_x', 'sentimiento_lm_x',
       'sentimiento_textblob_y', 'sentimiento_vader_y',
       'sentimiento_senticnet_y', 'sentimiento_lm_y','retorno','retorno_fijo', 'retorno clases','direccion'])

# writer = pd.ExcelWriter('OHLCV+indicadores+sentimientos_mod.xlsx', engine='xlsxwriter')
# for i in range(3):
#     dic_df_stock_final[empresas[i]].to_excel(writer, sheet_name=empresas[i])

writer.close()

# with open('dic_dfs_financiero_y_sentimientos.pkl', 'wb') as file:
#     pickle.dump(dic_df_stock_final, file)

### REGRESIÓN LOGISTICA DATOS DIARIOS
empresas = ['Ecopetrol', 'Bancolombia', 'Icolcap']
dic_df_stock_final = {}
dic_df_stock_final_scaled_std = {}
# dic_extenso_lg = {'Ecopetrol':{'df':dic_df_stock_final['Ecopetrol']}, 'Bancolombia':{'df':dic_df_stock_final['Bancolombia']},'Icolcap' :{'df':dic_df_stock_final['Icolcap']} }
for i in range(len(empresas)):

    dic_df_stock_final[empresas[i]] = pd.read_excel('OHLCV+indicadores+sentimientos_mod.xlsx', sheet_name=empresas[i], index_col=0)
    # dic_df_stock_final[empresas[i]] = dic_df_stock_final[empresas[i]][(dic_df_stock_final[empresas[i]].iloc[:,12] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,13] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,14] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,15] != 0) ]
    # ESTANDARIZACIÓN FEATURES

    scaler_std = StandardScaler()

    dic_df_stock_final_scaled_std[empresas[i]] = dic_df_stock_final[empresas[i]].copy()
    dic_df_stock_final_scaled_std[empresas[i]] = dic_df_stock_final_scaled_std[empresas[i]].drop(columns=['retorno clases', 'retorno_fijo'])

    dic_df_stock_final_scaled_std[empresas[i]]['retorno_lag_2' ] = dic_df_stock_final_scaled_std[empresas[i]]['retorno'].shift(1)
    dic_df_stock_final_scaled_std[empresas[i]] = dic_df_stock_final_scaled_std[empresas[i]].reindex(columns=['days_of_week', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'williams %R', 'mfi', 'macd(12-26)', 'atr(14)', 'adx(14)', 'textblob_1', 'vader_1', 'senticnet_1', 'lm_1', 'textblob_2', 'vader_2', 'senticnet_2', 'lm_2', 'retorno','retorno_lag_2', 'direccion' ])
    dic_df_stock_final_scaled_std[empresas[i]].iloc[:, 1:-1] = scaler_std.fit_transform(dic_df_stock_final_scaled_std[empresas[i]].iloc[:, 1:-1])
    # corr = dic_df_stock_final_scaled_std[empresas[i]].iloc[:,[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].corr()
    # fig, ax = plt.subplots(figsize=(14,14))
    # ax.tick_params(labelsize=19)
    # sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.savefig(f'correlacion_{empresas[i]}_mod_2.png')
    ### CORRELACIÓN ENTRE VARIABLES (DEPENDIENTES E INDEPENDIENTE)
    #
    # dic_extenso_lg[empresas[i]][f'corr_lag_{ventana}'] = dic_df_stock_final_scaled_std[empresas[i]]
    # dic_extenso_lg[empresas[i]][f'corr_lag_{ventana}'].iloc[:,6:24] = dic_df_stock_final_scaled_std[empresas[i]].iloc[:,6:24].shift(ventana)
    # dic_extenso_lg[empresas[i]][f'corr_lag_{ventana}'] = dic_extenso_lg[empresas[i]][f'corr_lag_{ventana}'].dropna()
    # dic_extenso_lg[empresas[i]][f'corr_lag_{ventana}'] = dic_extenso_lg[empresas[i]][f'corr_lag_{ventana}'].iloc[:,6:25].corr()
    # fig = plt.figure()
    # plt.rcParams["figure.figsize"] = (14,14)
    # corr = dic_extenso_lg[empresas[i]][f'corr_lag_{ventana}']
    # sns.heatmap(corr,vmin=-1,
    #         vmax=1, cmap='coolwarm')
    # plt.savefig(rf'Graficas regresión logistica\{empresas[i]}\ventana de tiempo {ventana}\corr_lag_{ventana} ')





    # ### FEATURE VARIABLES INDICADORES TECNICOS ( TRAIN 0.8, TEST 0.2)
    #
    # X = dic_df_stock_final_lag1[empresas[0]].iloc[:1245, [6,7,8,13]]
    #
    # y = dic_df_stock_final_lag1[empresas[0]].iloc[:1245, -1]
    # X_test = dic_df_stock_final_lag1[empresas[0]].iloc[1246:, [6,7,8,13]]
    #
    # y_test = dic_df_stock_final_lag1[empresas[0]].iloc[1246:, -1]
    #
    # sm_lg = sm.Logit(y,X)
    # result = sm_lg.fit()
    # result.summary()
    #
    # lg = LogisticRegression()
    # lg.fit(X,y)
    # lg_p = lg.predict(X_test)
    # acc = accuracy_score(y_test, lg_p)
    # roc_auc = roc_auc_score(y_test, lg_p)
    #
    # ### DATOS ESTANDARIZADOS
    # scaler_std = StandardScaler()
    # dic_df_stock_final_scaled_std = {}
    # dic_df_stock_final_scaled_std[empresas[0]] = dic_df_stock_final[empresas[0]].copy()
    # dic_df_stock_final_scaled_std[empresas[0]].iloc[:,1:23] = scaler_std.fit_transform(dic_df_stock_final_scaled_std[empresas[0]].iloc[:,1:23])
    #
    # ### CORRELACIÓN ENTRE VARIABLES ESTANDARIZADAS (DEPENDIENTES E INDEPENDIENTE) REDICCION DÍA T
    #
    # dic_corr_std = {}
    #
    # dic_corr_std['corr_lag_0_'+empresas[0]] = dic_df_stock_final_scaled_std[empresas[0]].iloc[:,6:23].corr()
    # f_std, ax_std = plt.subplots(figsize=(14,14))
    # ax_std = sns.heatmap(dic_corr_std['corr_lag_0_'+empresas[0]], cmap='coolwarm', vmax=1, vmin=-1)
    # plt.show()
    #
    ### SELECCIÓN DE CARACTERÍSTICAS L1 LASSO COEFICIENTE, SELECCIÓN DE CARACTERISTICAS DE FRECUENCIA (ITERACIONES POR FUERZA BRUTA)

    ### LASSO COEFFICIENT REDICCION DÍA T

    # from sklearn.linear_model import LassoCV
    #
    #
    # df = dic_df_stock_final_scaled_std[empresas[i]]
    # X_df = df.iloc[:,6:24].shift(ventana)
    # X_df = X_df.dropna()
    # X = X_df.iloc[:round(X_df.shape[0] * 0.8), :]
    # y = df.iloc[ventana:round(X_df.shape[0] * 0.8), -1]
    #
    # X_test = X_df.iloc[round(X_df.shape[0] * 0.8)+1:,:]
    # y_test = df.iloc[ventana + round(X_df.shape[0] * 0.8)+2:, -1]
    #
    #
    # lasso = LassoCV().fit(X, y)
    # importance = np.abs(lasso.coef_)
    # feature_names = np.array(dic_df_stock_final_scaled_std[empresas[i]].iloc[:, 6:24].columns)
    # plt.bar(height=importance, x=feature_names)
    # plt.xticks(rotation='vertical')
    # plt.title(f"Feature importances via coefficients")
    #
    # plt.rcParams["figure.figsize"] = (14, 14)
    # plt.show()
    # break
    # plt.savefig(rf'Graficas regresión logistica\{empresas[i]}\ventana de tiempo {ventana}\Lasso_coef_lag_{ventana} ')
    #
    #
    #
    #
    #
    # ### SELECCIÓN DE CARACTERISTICAS DE FRECUENCIA (ITERACIONES POR FUERZA BRUTA) PREDICCION DÍA T
    # from sklearn.feature_selection import SequentialFeatureSelector
    #
    #
    # sfs_forward = SequentialFeatureSelector(lasso, n_features_to_select=4,
    #                                     direction='forward').fit(X, y)
    #
    # print("Features selected by forward sequential selection: "
    #   f"{feature_names[sfs_forward.get_support()]}")

    #
    # ### PREDICCION DIA t (estandarizado)
    #
    # X = dic_df_stock_final_scaled_std[empresas[0]].iloc[:1245, 14:22]
    #
    # y = dic_df_stock_final_scaled_std[empresas[0]].iloc[:1245, -1]
    # X_test = dic_df_stock_final_scaled_std[empresas[0]].iloc[1246:, 14:22]
    #
    # y_test = dic_df_stock_final_scaled_std[empresas[0]].iloc[1246:, -1]
    #
    # sm_lg_s = sm.Logit(y,X)
    # result_s = sm_lg_s.fit()
    # result_s.summary()
    #
    # lg_s = LogisticRegression()
    # lg_s.fit(X,y)
    # lg_p_s = lg_s.predict(X_test)
    # acc_s = accuracy_score(y_test, lg_p_s)
    # roc_auc_s = roc_auc_score(y_test, lg_p_s)
    #
    # ### PREDICCION DIA t+1 (estandarizado)
    # dic_df_stock_final_lag1_s = {}
    # dic_df_stock_final_lag1_s[empresas[0]] = dic_df_stock_final_scaled_std[empresas[0]].copy()
    # dic_df_stock_final_lag1_s[empresas[0]].iloc[:,6:23] = dic_df_stock_final_scaled_std[empresas[0]].iloc[:,6:23].shift(1)
    # dic_df_stock_final_lag1_s[empresas[0]] = dic_df_stock_final_lag1_s[empresas[0]].dropna()
    #
    # ### FEATURE VARIABLES INDICADORES TECNICOS ESTANDARIZADOS ( TRAIN 0.8, TEST 0.2)
    #
    # X = dic_df_stock_final_lag1_s[empresas[0]].iloc[:1245, [6,7,8,9,10,11,12,13,18,22]]
    #
    # y = dic_df_stock_final_lag1_s[empresas[0]].iloc[:1245, -1]
    # X_test = dic_df_stock_final_lag1_s[empresas[0]].iloc[1246:, [6,7,8,9,10,11,12,13,18,22]]
    #
    # y_test = dic_df_stock_final_lag1_s[empresas[0]].iloc[1246:, -1]
    #
    # sm_lg_s = sm.Logit(y,X)
    # result_s = sm_lg_s.fit()
    # result_s.summary()
    #
    # lg_s = LogisticRegression()
    # lg_s.fit(X,y)
    # lg_p_s = lg_s.predict(X_test)
    # acc_s = accuracy_score(y_test, lg_p_s)
    # roc_auc_s = roc_auc_score(y_test, lg_p_s)


# lg = LogisticRegression(C= 0.01, penalty='l2', solver='liblinear')
# lg = LogisticRegression(C = 0.5455594781168515, penalty= 'l1', solver ='liblinear')
# prediccion = result.predict(X_test)
# param_grid = {"C":np.linspace(0.01, 1, 100), "penalty":['l2'], "solver": ['newton-cg', 'lbfgs', 'liblinear'] }

#
# param_grid = {}
# C = [0.00001, 0.0001, 0.001,0.1,1,10,100,1000]
# penalty = ['l2']
# solver = ['newton-cg', 'lbfgs', 'liblinear']
# i = 0
# for c in C:
#     for p in penalty:
#         for s in solver:
#             lg_pg = LogisticRegression(C=c, penalty=p, solver=s)
#             lg_pg.fit(X,y)
#             lg_p = lg_pg.predict(X_test)
#             cr = classification_report(y_test,lg_p, output_dict=True)
#             i+=1
#             param_grid['lg '+str(i)] = [c,p,s,cr['accuracy']]
#             print(param_grid['lg '+str(i)])






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

from sklearn.linear_model import LassoCV
# [6,7,8,9,10,,11,12,13,14,15,16,17,18,19,20,21,22] icolcap  0.5364156102411136
# [7,8,9,18]
# [16,17,19]  Ecopetrol 0.47003311258278146
# [10,16,17] Ecopetrol 0.47553807947019866
# [6,7,8,9,10,11] icolcap 0.55
# [7,12,13]

# df = dic_df_stock_final_scaled_std[empresas[2]].copy()
# df = df.drop(columns=['retorno clases','retorno_fijo'])
# for i in range(1,6):
#     df['retorno_lag_'+str(i)] = df['retorno'] - df['retorno'].shift(i)
# lista_idx = df.columns.to_list()
# lista_idx.insert(len(lista_idx)+1, lista_idx[-6])
# del(lista_idx[-7])
# lista_idx.insert(len(lista_idx)-1, lista_idx[-7])
# del(lista_idx[-8])
# df = df.reindex(columns=lista_idx)

# k= 7
# colmunas_moviles = list(range(7,11))
# for l,i  in enumerate(range(1,4)):
#     k = colmunas_moviles[l]
#     df.insert(loc=k, column='rsi_lag_'+str(i), value=0)
#     df['rsi_lag_'+str(i)] = df['rsi'].shift(i)
#     k+=2 +l
#     df.insert(loc=k, column='williams %R_lag_' + str(i), value=0)
#     df['williams %R_lag_' + str(i)] = df['williams %R'].shift(i)
#     k += 2 +l
#     df.insert(loc=k, column='mfi_lag_' + str(i), value=0)
#     df['mfi_lag_' + str(i)] = df['mfi'].shift(i)
#     k += 2+l
#     df.insert(loc=k, column='macd(12-26)_lag_' + str(i), value=0)
#     df['macd(12-26)_lag_' + str(i)] = df['macd(12-26)'].shift(i)
#     k += 2 +l
#     df.insert(loc=k, column='atr(14)_lag_' + str(i), value=0)
#     df['atr(14)_lag_' + str(i)] = df['atr(14)'].shift(i)
#     k += 2 + l
#     df.insert(loc=k, column='adx(14)_lag_' + str(i), value=0)
#     df['adx(14)_lag_' + str(i)] = df['adx(14)'].shift(i)







# df= df.drop(columns=df.columns.to_list()[30:38])

# df.iloc[:, 6:47]
df = dic_df_stock_final_scaled_std[empresas[2]].copy()
df.iloc[:,6:22] = df.iloc[:,6:22].shift(1)
df = df.dropna()

# # df.iloc[10,


X = df.iloc[:round(df.shape[0] * 0.8),12:20]
y = df.iloc[:round(df.shape[0] * 0.8), -1]

X_test = df.iloc[round(df.shape[0] * 0.8):,12:20]
y_test = df.iloc[round(df.shape[0] * 0.8):, -1]

sm_lg_s = sm.Logit(y,X)
result_s = sm_lg_s.fit()





lg_s = LogisticRegression()
lg_s.fit(X,y)
lg_p_s = lg_s.predict(X_test)
print(accuracy_score(y_test, lg_p_s), '\n'*2, roc_auc_score(y_test, lg_p_s))



lasso = LassoCV(max_iter=1000).fit(X, y)
importance = sorted(np.abs(lasso.coef_),reverse=True)

feature_names = np.array(df.iloc[:,6:12].columns)
for i , j in enumerate(importance):
    if importance[i] != 0:
        print(f'primer feature {feature_names[i]}: {importance[i]}')
        if i > 2:
            break
    else: break
plt.bar(height=importance, x=feature_names)
plt.xticks(rotation='vertical')
plt.title(f"Feature importances via coefficients")

plt.rcParams["figure.figsize"] = (6, 6)
plt.show()




from sklearn.feature_selection import SequentialFeatureSelector


sfs_forward = SequentialFeatureSelector(lasso, n_features_to_select=3,
                                    direction='forward').fit(X, y)

print("Features selected by forward sequential selection: "
  f"{feature_names[sfs_forward.get_support()]}")

sfs_forward = SequentialFeatureSelector(lasso, n_features_to_select=3,
                                    direction='backward').fit(X, y)

print("Features selected by backward sequential selection: "
  f"{feature_names[sfs_forward.get_support()]}")




#
# dic_df_stock_final[empresas[0]] = dic_df_stock_final[empresas[0]][dic_df_stock_final[empresas[0]]['days_of_week'] == 4]
#
# ####
#
# ### REGRESIÓN LOGISTICA DATOS SEMANALES
# empresas = ['Ecopetrol', 'Bancolombia', 'Icolcap']
# dic_df_stock_final = {}
# dic_df_stock_final_scaled_std = {}
# # dic_extenso_lg = {'Ecopetrol':{'df':dic_df_stock_final['Ecopetrol']}, 'Bancolombia':{'df':dic_df_stock_final['Bancolombia']},'Icolcap' :{'df':dic_df_stock_final['Icolcap']} }
# for i in range(len(empresas)):
#
#     dic_df_stock_final[empresas[i]] = pd.read_excel('OHLCV+indicadores+sentimientos_mod.xlsx', sheet_name=empresas[i], index_col=0)
#     # dic_df_stock_final[empresas[i]] = dic_df_stock_final[empresas[i]][(dic_df_stock_final[empresas[i]].iloc[:,12] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,13] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,14] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,15] != 0) ]
#     # ESTANDARIZACIÓN FEATURES
#
#     scaler_std = StandardScaler()
#
#
#     dic_df_stock_final[empresas[i]] = dic_df_stock_final[empresas[i]][
#         (dic_df_stock_final[empresas[i]].iloc[:, 12] != 0) & (dic_df_stock_final[empresas[i]].iloc[:, 13] != 0) & (
#                     dic_df_stock_final[empresas[i]].iloc[:, 14] != 0) & (
#                     dic_df_stock_final[empresas[i]].iloc[:, 15] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,16] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,17] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,18] != 0) & (dic_df_stock_final[empresas[i]].iloc[:,19] != 0)
# ]
#     dic_df_stock_final_scaled_std[empresas[i]] = dic_df_stock_final[empresas[i]].copy()
#     dic_df_stock_final_scaled_std[empresas[i]]['retorno'] = dic_df_stock_final_scaled_std[empresas[i]]['close']-dic_df_stock_final_scaled_std[empresas[i]]['open']
#     # dic_df_stock_final_scaled_std[empresas[i]].iloc[:, 1:-2] = scaler_std.fit_transform(dic_df_stock_final_scaled_std[empresas[i]].iloc[:, 1:-2])
#     corr = dic_df_stock_final_scaled_std[empresas[i]].iloc[:,[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].corr()
#     fig, ax = plt.subplots(figsize=(14,14))
#     ax.tick_params(labelsize=19)
#     sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=True)
#     plt.savefig(f'correlacion_{empresas[i]}_dep_2.png')