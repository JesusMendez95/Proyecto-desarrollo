import numpy as np
import pandas as pd

np.random.seed(123)

# columns = ['a', 'b', 'c']
#
# data = np.random.randint(0, 100, (3, 2))
# lis = [1,2,3,4]
# dic = [dict({'variable{}'.format(i):j}) for i,j in enumerate(lis)]

# for x in range(9):
#     exec("string" + str(x) + " = []")

ecopetrol_df_indicadores = pd.read_excel('Ecopetrol OHLCV+indicadores.xlsx', index_col='date')
bancolombia_df_indicadores = pd.read_excel('Bancolombia OHLCV+indicadores.xlsx', index_col='date')
icolcap_df_indicadores = pd.read_excel('Icolcap OHLCV+indicadores.xlsx', index_col='date')


def ajuste_df(df):
    df_adjusted = pd.DataFrame()
    df_adjusted['Precio de Cierre'] = df.close
    df_adjusted['Retorno'] = df.close - df.close.shift(1)
    df_adjusted['Delta alto-bajo'] = df.high - df.low
    df_adjusted['Volumen'] = df.volume / 1000000
    return df_adjusted


df_ajustado_eco = ajuste_df(ecopetrol_df_indicadores)
df_ajustado_ban = ajuste_df(bancolombia_df_indicadores)
df_ajustado_col = ajuste_df(icolcap_df_indicadores)


def summary_statistics(*dfs):
    tuplas = [('Precio', 'min'), ('Precio', 'max'), ('Precio', 'media'), ('Precio', 'std'), ('Retorno', 'min'),
              ('Retorno', 'max'), ('Retorno', 'media'), ('Retorno', 'std'), ('Delta alto-bajo', 'min'),
              ('Delta alto-bajo', 'max'), ('Delta alto-bajo', 'media'), ('Delta alto-bajo', 'std'), ('Volumen', 'min'),
              ('Volumen', 'max'), ('Volumen', 'media'), ('Volumen', 'std')]
    indice = pd.MultiIndex.from_tuples(tuplas, names=['Variable', 'Estadistico'])
    columnas = ['Ecopetrol', 'Bancolombia', 'Icolcap']
    rng = np.random.default_rng(0)
    dummy_array = rng.random((16, 3))
    df_ss = pd.DataFrame(data=dummy_array, index=indice, columns=columnas)
    index_count = 0
    for df in dfs:

        df_ss[columnas[index_count]] = [df['Precio de Cierre'].min(), df['Precio de Cierre'].max(),
                                        df['Precio de Cierre'].mean(), df['Precio de Cierre'].std(),
                                        df['Retorno'].min(), df['Retorno'].max(), df['Retorno'].mean(),
                                        df['Retorno'].std(),
                                        df['Delta alto-bajo'].min(), df['Delta alto-bajo'].max(),
                                        df['Delta alto-bajo'].mean(), df['Delta alto-bajo'].std(), df['Volumen'].min(),
                                        df['Volumen'].max(), df['Volumen'].mean(), df['Volumen'].std()]
        index_count += 1
    return df_ss


df_master = summary_statistics(df_ajustado_eco, df_ajustado_ban, df_ajustado_col)
df_ajustado_eco = df_ajustado_eco.dropna()
df_ajustado_col = df_ajustado_col.dropna()
df_ajustado_ban = df_ajustado_ban.dropna()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('darkgrid')
# fig, ax = plt.subplots(1,3)
# ax[0].boxplot(df_ajustado_eco['Precio de Cierre'])
# ax[0].set_xlabel('Ecopetrol')
# ax[1].boxplot(df_ajustado_ban['Precio de Cierre'], showfliers=False)
# ax[1].set_xlabel('Bancolombia')
# ax[2].boxplot(df_ajustado_col['Precio de Cierre'])
# ax[2].set_xlabel('Icolcap')
fig, ax = plt.subplots(1,3)
ax[0].boxplot(df_ajustado_eco['Volumen'])
ax[0].set_xlabel('Ecopetrol')
ax[1].boxplot(df_ajustado_ban['Volumen'])
ax[1].set_xlabel('Bancolombia')
ax[2].boxplot(df_ajustado_col['Volumen'])
ax[2].set_xlabel('Icolcap')

plt.hist(df_ajustado_col['Volumen'], bins= 30, color='yellow', alpha=0.3)
plt.hist(df_ajustado_eco['Volumen'], bins= 30, color='red', alpha= 0.3)
plt.hist(df_ajustado_ban['Volumen'], bins= 15, color='blue', alpha= 0.3)
plt.show()
# data_list = [df_ajustado_eco['Precio de Cierre'], df_ajustado_ban['Precio de Cierre'], df_ajustado_col['Precio de Cierre']]
plt.boxplot(df_ajustado_eco['Precio de Cierre'])
plt.boxplot(df_ajustado_ban['Precio de Cierre'])
plt.show()
import xlsxwriter
import xlwt
writer = pd.ExcelWriter('summary_statistics.xlsx', engine='xlsxwriter')
vol_ss = pd.DataFrame([df_ajustado_eco['Volumen'].describe(),df_ajustado_ban['Volumen'].describe(), df_ajustado_col['Volumen'].describe()],index=['Ecopetrol', 'Bancolombia', 'Icolcap'])
vol_ss_t= vol_ss.T
vol_ss_t.to_excel(writer, sheet_name='Volumen')

price_ss = pd.DataFrame([df_ajustado_eco['Precio de Cierre'].describe(),df_ajustado_ban['Precio de Cierre'].describe(), df_ajustado_col['Precio de Cierre'].describe()],index=['Ecopetrol', 'Bancolombia', 'Icolcap'])
price_ss_t= price_ss.T
price_ss_t.to_excel(writer, sheet_name='Precio de Cierre')
return_ss = pd.DataFrame([df_ajustado_eco['Retorno'].describe(),df_ajustado_ban['Retorno'].describe(), df_ajustado_col['Retorno'].describe()],index=['Ecopetrol', 'Bancolombia', 'Icolcap'])
return_ss_t= return_ss.T
return_ss_t.to_excel(writer, sheet_name='Retorno')
delta_ss = pd.DataFrame([df_ajustado_eco['Delta alto-bajo'].describe(),df_ajustado_ban['Delta alto-bajo'].describe(), df_ajustado_col['Delta alto-bajo'].describe()],index=['Ecopetrol', 'Bancolombia', 'Icolcap'])
delta_ss_t= delta_ss.T
delta_ss_t.to_excel(writer, sheet_name='Delta alto-bajo')
writer.save()