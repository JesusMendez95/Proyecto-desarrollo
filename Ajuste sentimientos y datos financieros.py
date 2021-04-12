import pickle
import pandas as pd
import datetime
import numpy as np


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

    dic_df_stock_final[empresas[i]] = df_time.merge(pd.read_excel('OHLCV+indicadores_mod.xlsx', index_col=0, sheet_name=f'{empresas[i]} OHLCV+indicadores'), right_index=True, left_index=True, how='left')
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
       'atr(14)','adx(7)', 'sentimiento_textblob_x',
       'sentimiento_vader_x', 'sentimiento_senticnet_x', 'sentimiento_lm_x',
       'sentimiento_textblob_y', 'sentimiento_vader_y',
       'sentimiento_senticnet_y', 'sentimiento_lm_y','retorno','retorno_fijo', 'retorno clases','direccion'])

writer = pd.ExcelWriter('OHLCV+indicadores+sentimientos_mod.xlsx', engine='xlsxwriter')
for i in range(3):
    dic_df_stock_final[empresas[i]].to_excel(writer, sheet_name=empresas[i])

writer.close()


### CREACIÓN DATASETS SEMANALES (CADA OBSERVACIÓN ES EL ÚLTIMO DIA BURSATIL DE LA SEMANA)

empresas = ['Ecopetrol', 'Bancolombia', 'Icolcap']
def pred_semanal():
    excel_writer = pd.ExcelWriter('OHLCV+indicadores+sentimientos_semanal_mod.xlsx', engine='xlsxwriter')
    for index in range(len(empresas)):
        df = pd.read_excel('OHLCV+indicadores+sentimientos_mod.xlsx', sheet_name=empresas[index] , index_col=1)
        dic_test = {}

        j = 0
        for i in range(len(df)-1):
            dic_test.setdefault(f'semana{j}',[])
            if df.index[i+1] - df.index[i] == 1:
                dic_test[f'semana{j}'].append(df.iloc[i, 12:20])
            elif  df.index[i+1] - df.index[i] == 2 :
                dic_test[f'semana{j}'].append(df.iloc[i, 12:20])
            else:
                dic_test[f'semana{j}'].append(df.iloc[i, 12:20])
                j+=1

        lista=[]

        for key, value in dic_test.items():
            lista.append(sum(value)/len(value))

        df_copy = df.copy()

        ultimo_dia_semana= []
        k = 0
        for i in range(len(df)-1):
            if df.index[i+1] - df.index[i] == 1 or df.index[i+1] - df.index[i] == 2:
                continue
            else:
                df_copy.iloc[i,12:20] = lista[k]
                ultimo_dia_semana.append(df_copy.iloc[i,:])
                k += 1
        df_semanal = pd.DataFrame(data=np.zeros(shape= (len(ultimo_dia_semana),len(df.columns))) ,columns=df.columns)

        for i in range(len(df_semanal)):
            df_semanal.iloc[i,:] = ultimo_dia_semana[i]

        df_semanal = df_semanal.set_index('index')
        df_semanal['retorno'] = df_semanal['close'] - df_semanal['close'].shift(1)
        df_semanal = df_semanal.dropna()
        df_semanal['direccion'] = df_semanal['retorno'].apply(lambda x: 1 if x>0 else 0)
        df_semanal.to_excel(excel_writer, sheet_name=empresas[index])
    excel_writer.close()

pred_semanal()

