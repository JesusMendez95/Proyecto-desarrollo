# Ajuste datos financieros


import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pmdarima import model_selection
import pmdarima as pm
import technical_indicators_lib
import seaborn as sns

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns',
              1000)  # Setup para ampliar visualización de datos (en este caso DataFrame) en el IDE
pd.set_option('display.width', 1000)
ecopetrol_df = pd.read_excel('Ecopetrol 2013-2019 raw data.xls')
bancolombia_df = pd.read_excel('Bancolombia 2013-2019 raw data.xls')
colcap_df = pd.read_csv(
    "Icolcap 2013-2019 raw data.csv")  # Importar datos financieros como Dataframe, (previamente convertidos de


# Formato xls a csv) usando pandas

def serie_financiera(stock_df):
    stock_df['Precio Apertura'] = stock_df['Precio Cierre'] - stock_df['Variacion Absoluta']
    # Se calcula el valor "Open" de la accion

    columns_fixed = ['fecha', 'Precio Apertura', 'Precio Mayor', 'Precio Menor', 'Precio Cierre', 'Volumen']
    stock_df = stock_df[columns_fixed]

    stock_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    # Orden de las columnas de datos, según formato requerido

    high_list = []
    low_list = []
    for index, row in stock_df.iterrows():
        if row['open'] > row['high']:
            high_list.append(row.open)
        else:
            high_list.append(row.high)
        if row['open'] < row['low']:
            low_list.append(row.open)
        else:
            low_list.append(row.low)

    stock_df['high'] = high_list
    stock_df['low'] = low_list

    # stock_df['date'] = stock_df['date'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'))

    def df_to_float(df):
        for i in range(len(df.columns)):
            if i == 0:
                continue
            df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], downcast='float')

    df_to_float(stock_df)

    stock_df = stock_df.set_index('date')

    # stock_df_xlsx = stock_df.to_excel('Icolcap OHLCV.xlsx', encoding='utf-8')
    # Conversión y guardado de los datos en disco
    high_list = []
    low_list = []

    return stock_df


ecopetrol_df = serie_financiera(ecopetrol_df)
bancolombia_df = serie_financiera(bancolombia_df)
colcap_df = serie_financiera(colcap_df)

ecopetrol_df_indicadores = pd.read_excel('Ecopetrol OHLCV+indicadores.xlsx', index_col='date')
bancolombia_df_indicadores = pd.read_excel('Bancolombia OHLCV+indicadores.xlsx', index_col='date')
icolcap_df_indicadores = pd.read_excel('Icolcap OHLCV+indicadores.xlsx', index_col='date')

# Graficar serie


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
    df_adjusted['Delta alto-bajo'] = df.high - df.low
    df_adjusted['Volumen'] = df.volume
    return df_adjusted


ajuste_df(ecopetrol_df_indicadores)
df = pd.DataFrame()
datos = []
datos.append()

ss_df = pd.DataFrame(example, columns=ss2_columns, index=index)
print(ss_df)

# plt.xlabel('Año')
# plt.ylabel('Precio (COP)')
# plt.vlines(pd.Timestamp('2016-01-01'), 0, stock_df.close.max(), linestyle='dashed', color='red', alpha=0.5)
# plt.vlines(pd.Timestamp('2017-10-01'), 0, stock_df.close.max(), linestyle='dashed', color='red', alpha=0.5)


# Notas La hipótesis nula de Augmented Dickey-Fuller es que existe una raíz unitaria, con la alternativa de que no
# existe una raíz unitaria. Si el valor p está por encima de un tamaño crítico, no podemos rechazar que haya una raíz
# unitaria. Los valores p se obtienen mediante la aproximación de la superficie de regresión de MacKinnon 1994,
# pero utilizando las tablas actualizadas de 2010. Si el valor p está cerca de ser significativo, entonces los
# valores críticos deben usarse para juzgar si rechazar el nulo. La opción autolag y maxlag se describen en Greene.

# Determinar si la serie es estacionaria

# def test_estacionariedad(serie):
#     # Metodo gráfico, Medidas estadisticas moviles:
#     media_movil = serie.rolling(window=10).mean()
#     std_movil = serie.rolling(window=10).std()
#
#     # Graficar estadisticas moviles:
#
#     original = plt.plot(serie, color='blue', label='Original')
#     media = plt.plot(media_movil, color='red', label='Media Movil')
#     desviacion = plt.plot(std_movil, color='black', label='Desviación Estantar Movil')
#     plt.legend(loc='best')
#     plt.title('Estadisticas Moviles')
#
#     # Graficar Funcion Autocorrelación
#     plot_acf(serie)
#
#     # Graficar Función Autocorrelación Parcial
#     plot_pacf(serie)
#     plt.show()
#
#     # Test Augmented Dickey-Fuller:
#     print('Resultados del Test ADK:')
#     adk_test = adfuller(serie, autolag='AIC')
#     resultado_adk = pd.Series(adk_test[0:4], index=['Test Estadistico', 'p-value', '#Rezagos usados',
#                                                     'Numero de Observaciones Usadas'])
#     for key, value in adk_test[4].items():
#         resultado_adk['Valor critico {}%'.format(key)] = value
#     print(resultado_adk)
#     if adk_test[1] >= 0.05:
#         print('Se rechaza Ho, hay evidencia significativa de que la serie no tiene raiz unitaria')
#     else:
#         print('No se rechaza Ho, hay evidencia significativa de que la serie  tiene raiz unitaria')


# Diferenciacion


"""s_p_cierre_diff1 = serie_p_cierre - serie_p_cierre.shift(1)
s_p_cierre_diff1 = s_p_cierre_diff1.dropna()

train, test = model_selection.train_test_split(serie_p_cierre.iloc[0:200,0], train_size=50)

arima = pm.auto_arima(train, trace=True)


# Plot actual test vs. forecasts:
x = np.arange(test.shape[0])
plt.plot(x, test)
plt.plot(x, arima.predict(n_periods=test.shape[0]), color='red', alpha=0.5)
plt.title('Actual test samples vs. forecasts')
plt.show()"""

"print(arima.summary())"
