import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
import math




# Metricas generales

empresas = ['Ecopetrol','Bancolombia', 'Icolcap']
lags = [0,1,2,3]
dic_metrics_completo = {}
dic_metrics_titulo = {}
dic_metrics = {}
clasificadores = ['sentimiento_textblob_x', 'sentimiento_vader_x', 'sentimiento_senticnet_x', 'sentimiento_lm_x','sentimiento_textblob_y', 'sentimiento_vader_y', 'sentimiento_senticnet_y', 'sentimiento_lm_y']
for i in range(len(empresas)):
    df_empresa = pd.read_excel('OHLCV+indicadores+sentimientos_mod.xlsx', sheet_name=empresas[i], index_col=0)
    for l in lags:
        df_empresa.iloc[:,12:20] = df_empresa.iloc[:,12:20].shift(l)

        for j in range(len(clasificadores)):
            dummy = df_empresa.loc[:, ['retorno clases', clasificadores[j]]].dropna()

            dummy.loc[:,clasificadores[j]] = dummy[clasificadores[j]].apply(lambda x: x/math.sqrt(abs(x**2)) if x !=0 else 0)
            dummy.loc[:, clasificadores[j]] = dummy[clasificadores[j]].apply(lambda x: float('NaN') if x == 0 else x)
            dummy = dummy.dropna()
            dic_metrics['accuracy_score_' + empresas[i] + '_' + clasificadores[j]+ f'_lag{l}'] = accuracy_score(
                dummy['retorno clases'], dummy[clasificadores[j]])
            dic_metrics['roc_auc_score_' + empresas[i] + '_' + clasificadores[j]+ f'_lag{l}'] = roc_auc_score(dummy['retorno clases'],dummy[clasificadores[j]])

metricas_lag_0 = []
for key, value in dic_metrics.items():
    if int(key[-1]) == 0:
        metricas_lag_0.append(value)
    else: continue

metricas_lag_1 = []
for key, value in dic_metrics.items():
    if int(key[-1]) == 1:
        metricas_lag_1.append(value)
    else:
        continue

metricas_lag_2 = []
for key, value in dic_metrics.items():
    if int(key[-1]) == 2:
        metricas_lag_2.append(value)
    else:
        continue

metricas_lag_3 = []
for key, value in dic_metrics.items():
    if int(key[-1]) == 3:
        metricas_lag_3.append(value)
    else:
        continue

df_acc_auc_roc_lag_0 = pd.DataFrame(data=np.array(metricas_lag_0).reshape(24,2), columns=['accuracy_score','auc_roc_score'])
df_acc_auc_roc_lag_1 = pd.DataFrame(data=np.array(metricas_lag_1).reshape(24,2), columns=['accuracy_score','auc_roc_score'])
df_acc_auc_roc_lag_2 = pd.DataFrame(data=np.array(metricas_lag_2).reshape(24,2), columns=['accuracy_score','auc_roc_score'])
df_acc_auc_roc_lag_3 = pd.DataFrame(data=np.array(metricas_lag_3).reshape(24,2), columns=['accuracy_score','auc_roc_score'])

dic_acc_auc_roc_lag_mean = dict(
    accuracy_score=[df_acc_auc_roc_lag_0['accuracy_score'].min(),df_acc_auc_roc_lag_0['accuracy_score'].mean(),
                    df_acc_auc_roc_lag_0['accuracy_score'].max(),df_acc_auc_roc_lag_1['accuracy_score'].min(),df_acc_auc_roc_lag_1['accuracy_score'].mean(),
                    df_acc_auc_roc_lag_1['accuracy_score'].max(),df_acc_auc_roc_lag_2['accuracy_score'].min(),df_acc_auc_roc_lag_2['accuracy_score'].mean(),
                    df_acc_auc_roc_lag_2['accuracy_score'].max(),df_acc_auc_roc_lag_3['accuracy_score'].min(),df_acc_auc_roc_lag_3['accuracy_score'].mean(),
                    df_acc_auc_roc_lag_3['accuracy_score'].max()],
    auc_roc_score=[df_acc_auc_roc_lag_0['auc_roc_score'].min(),df_acc_auc_roc_lag_0['auc_roc_score'].mean(),
                    df_acc_auc_roc_lag_0['auc_roc_score'].max(),df_acc_auc_roc_lag_1['auc_roc_score'].min(),df_acc_auc_roc_lag_1['auc_roc_score'].mean(),
                    df_acc_auc_roc_lag_1['auc_roc_score'].max(),df_acc_auc_roc_lag_2['auc_roc_score'].min(),df_acc_auc_roc_lag_2['auc_roc_score'].mean(),
                    df_acc_auc_roc_lag_2['auc_roc_score'].max(),df_acc_auc_roc_lag_3['accuracy_score'].min(),df_acc_auc_roc_lag_3['auc_roc_score'].mean(),
                    df_acc_auc_roc_lag_3['auc_roc_score'].max()])

lista_multi_index = []
valores = ['valor_min','valor_medio','valor_max']
for i in range(4):
    lista_multi_index.extend([(f'predicción_día_{i}', valores[0]),(f'predicción_día_{i}',valores[1]),(f'predicción_día_{i}',valores[2])])

index= pd.MultiIndex.from_tuples(lista_multi_index, names=['Día predicción','Estadístico'])

df_acc_auc_roc = pd.DataFrame(index=index, data=dic_acc_auc_roc_lag_mean, columns=['accuracy_score','auc_roc_score'])

df_acc_auc_roc.to_excel('metricas sentimientos resagados_mod.xlsx')


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns',
              1000)  # Setup para ampliar visualización de datos (en este caso DataFrame) en el IDE
pd.set_option('display.width', 1000)
df_metricas = pd.read_excel('df_metricas.xlsx', index_col=[0, 1], header=[0, 1])

print(df_metricas.iloc[[0, 1], :])

df_metricas = df_metricas.T

# Metricas generales
dic_metrics_completo = {}
dic_metrics_titulo = {}
dic_metrics_pe = {}
import math
clasificadores = ['sentimiento_textblob_x', 'sentimiento_vader_x', 'sentimiento_senticnet_x', 'sentimiento_lm_x','sentimiento_textblob_y', 'sentimiento_vader_y', 'sentimiento_senticnet_y', 'sentimiento_lm_y']

for i in range(len(empresas)):
    df_empresa = pd.read_excel('OHLCV+indicadores+sentimientos_mod.xlsx', sheet_name=empresas[i], index_col=0)
    for j in range(len(clasificadores)):
        dummy = df_empresa.loc[:, ['retorno clases', clasificadores[j]]].dropna()

        dummy.loc[:, clasificadores[j]] = dummy[clasificadores[j]].apply(
            lambda x: x / math.sqrt(abs(x ** 2)) if x != 0 else 0)
        dummy.loc[:, clasificadores[j]] = dummy[clasificadores[j]].apply(lambda x: float('NaN') if x == 0 else x)
        dummy = dummy.dropna()

        dic_metrics_pe['confusion_matrix_' + empresas[i]+'_'+clasificadores[j]] = confusion_matrix(dummy['retorno clases'],dummy[clasificadores[j]])
        dic_metrics_pe['classification_report_'+empresas[i]+'_'+clasificadores[j]] = classification_report(dummy['retorno clases'],dummy[clasificadores[j]], output_dict=True)

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns',
              1000)  # Setup para ampliar visualización de datos (en este caso DataFrame) en el IDE
pd.set_option('display.width', 1000)



df_metricas = pd.read_excel('df_metricas.xlsx', index_col=[0,1], header =[0,1])
df_metricas = df_metricas.T
empresas = ['Ecopetrol', 'Bancolombia', 'Icolcap']
k = 0
for i in range(len(empresas)):

    for j in range(len(clasificadores)):

        df_metricas.iloc[0,k] = dic_metrics_pe['classification_report_'+empresas[i]+'_'+clasificadores[j]]['accuracy']
        df_metricas.iloc[1, k] = dic_metrics_pe['classification_report_' + empresas[i] + '_' + clasificadores[j]][
            '1']['precision']
        df_metricas.iloc[2, k] = dic_metrics_pe['classification_report_' + empresas[i] + '_' + clasificadores[j]][
            '-1']['precision']
        df_metricas.iloc[3, k] = dic_metrics_pe['classification_report_' + empresas[i] + '_' + clasificadores[j]][
            '1']['recall']
        df_metricas.iloc[4, k] = dic_metrics_pe['classification_report_' + empresas[i] + '_' + clasificadores[j]][
            '-1']['recall']
        df_metricas.iloc[5, k] = dic_metrics_pe['classification_report_' + empresas[i] + '_' + clasificadores[j]][
            'weighted avg']['f1-score']
        df_metricas.iloc[6, k] = dic_metrics_pe['classification_report_' + empresas[i] + '_' + clasificadores[j]][
            '1']['support']
        df_metricas.iloc[7, k] = dic_metrics_pe['classification_report_' + empresas[i] + '_' + clasificadores[j]][
            '-1']['support']


        k+=1


df_metricas = df_metricas.T

df_metricas.to_excel('metricas_clasificadores_mod_4.xlsx', sheet_name='Sheet2')


