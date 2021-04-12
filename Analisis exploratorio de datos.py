import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords



ecopetrol_df_indicadores = pd.read_excel('Ecopetrol OHLCV+indicadores.xlsx', index_col='date')
bancolombia_df_indicadores = pd.read_excel('Bancolombia OHLCV+indicadores.xlsx', index_col='date')
icolcap_df_indicadores = pd.read_excel('Icolcap OHLCV+indicadores.xlsx', index_col='date')


# EDA matplotlib


# Wordcloud
empresas = ['ecopetrol', 'bancolombia', 'colcap']
corpus_list = 0

for i in range(len(empresas)):
    empresa = pd.read_excel('noticias_larepublica_completo', sheet_name=empresas[i], index_col=0)
    corpus_list = [noticia for noticia in empresa['noticia']]
    corpus_str = ' '.join(corpus_list)
    text_cloud = WordCloud(width=2000, height=1000, background_color='white',
                           stopwords=stopwords.words('spanish')).generate(corpus_str)
    plt.figure(figsize=(20, 10))
    text_cloud.to_file(f'wordcloud_{empresas[i]}.png')
    plt.axis("off")
    plt.imshow(text_cloud, interpolation='bilinear')
    plt.show()

# GRAFICAR SERIES DE TIEMPO DE LAS ACCIONES

sns.set_style("whitegrid")
labels = ['ECOPETROL', 'BANCOLOMBIA', 'ICOLCAP']

fig, ax = plt.subplots(3, 1)
ax[0].plot(ecopetrol_df_indicadores.close, color='green', label='ECOPETROL')
ax[0].set_ylabel('Precio (COP)', fontweight='bold')
ax[0].legend(loc='lower right')
ax[0].vlines(pd.Timestamp('2016-01-01'), ecopetrol_df_indicadores.close.min(), ecopetrol_df_indicadores.close.max(),
             linestyle='dashed', color='black', alpha=.5)
ax[0].vlines(pd.Timestamp('2017-11-03'), ecopetrol_df_indicadores.close.min(), ecopetrol_df_indicadores.close.max(),
             linestyle='dashed', color='black', alpha=.5)

ax[1].plot(bancolombia_df_indicadores.close, color='blue', label='BANCOLOMBIA')
ax[1].set_ylabel('Precio (COP)', fontweight='bold')
ax[1].legend(loc='lower right')
ax[1].vlines(pd.Timestamp('2016-01-01'), bancolombia_df_indicadores.close.min(), bancolombia_df_indicadores.close.max(),
             linestyle='dashed', color='black', alpha=.5)
ax[1].vlines(pd.Timestamp('2017-11-03'), bancolombia_df_indicadores.close.min(), bancolombia_df_indicadores.close.max(),
             linestyle='dashed', color='black', alpha=.5)
ax[2].plot(icolcap_df_indicadores.close, color='red', label='ICOLCAP')
ax[2].vlines(pd.Timestamp('2016-01-01'), icolcap_df_indicadores.close.min(), icolcap_df_indicadores.close.max(),
             linestyle='dashed', color='black', alpha=.5)
ax[2].vlines(pd.Timestamp('2017-11-03'), icolcap_df_indicadores.close.min(), icolcap_df_indicadores.close.max(),
             linestyle='dashed', color='black', alpha=.5)
ax[2].set_ylabel('Precio (COP)', fontweight='bold')
ax[2].set_xlabel('Año', fontweight='bold')
ax[2].legend(loc='lower right')

fig.suptitle('Precio de las acciones a través del tiempo', fontweight='bold')
plt.show()
ss_columns = ['Media', 'Desviación Estandar', 'Media', 'Desviación Estandar', 'Media', 'Desviación Estandar']
ss2_columns = ['Precio de Cierre', 'Delta Precio Mayor y Menor', 'Volumen']
ss_index = [['Ecopetrol', 'Ecopetrol', 'Bancolombia', 'Bancolombia', 'ICOLCAP', 'ICOLCAP'],
            ['Media', 'Desviación Estandar', 'Media', 'Desviación Estandar', 'Media', 'Desviación Estandar']]
tuplas = list(zip(*ss_index))
index = pd.MultiIndex.from_tuples(tuplas, names=['Acción', 'Estadistico'])

rng = np.random.default_rng(0)
example = rng.random((6, 3))
eco_precio_mean = ecopetrol_df_indicadores.close.mean()


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
