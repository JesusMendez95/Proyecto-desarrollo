# Ajuste datos financieros


import pandas as pd
import matplotlib.pyplot as plt
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

ecopetrol_df = pd.read_excel("Ecopetrol 2013-2019 raw data.xls")  # Importar datos financieros como Dataframe, (previamente convertidos de
# Formato xls a csv) usando pandas


ecopetrol_df['Precio Apertura'] = ecopetrol_df['Precio Cierre'] - ecopetrol_df['Variacion Absoluta']
# Se calcula el valor "Open" de la accion


columns_fixed = ['fecha', 'Precio Apertura', 'Precio Mayor', 'Precio Menor', 'Precio Cierre', 'Volumen']
ecopetrol_df = ecopetrol_df[columns_fixed]

ecopetrol_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
# Orden de las columnas de datos, según formato requerido

high_list = []
low_list = []
for index, row in ecopetrol_df.iterrows():
    if row['open'] > row['high']:
        high_list.append(row.open)
    else: high_list.append(row.high)
    if row['open'] < row['low']:
        low_list.append(row.open)
    else: low_list.append(row.low)

ecopetrol_df['high'] = high_list
ecopetrol_df['low'] = low_list

ecopetrol_df['date'] = pd.to_datetime(ecopetrol_df['date'], '%Y-%m-%d')


def df_to_float(df):
    for i in range(len(df.columns)):
        if i == 0:
            continue
        df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], downcast='float')


df_to_float(ecopetrol_df)

ecopetrol_df = ecopetrol_df.set_index('date')

ecopetrol_df_xlsx = ecopetrol_df.to_excel('Ecopetrol OHLCV.xlsx', encoding='utf-8')
# Conversión y guardado de los datos en disco










print(ecopetrol_df.head())


# Graficar serie
import seaborn as sns
sns.set_style("whitegrid")

plt.plot(ecopetrol_df.index, ecopetrol_df.close)

plt.title('Precio de la acción de ECOPETROL atraves del tiempo')
plt.xlabel('Año')
plt.ylabel('Precio (COP)')
plt.vlines(pd.Timestamp('2016-01-01'),0, ecopetrol_df.close.max(), linestyle='dashed', color='red', alpha=0.5)
plt.vlines(pd.Timestamp('2017-10-01'),0, ecopetrol_df.close.max(), linestyle='dashed', color='red', alpha=0.5)

plt.show()

import seaborn as sns
sns.set_style("whitegrid")

plt.plot(ecopetrol_df.index, ecopetrol_df.close)

plt.title('Precio de la acción de ECOPETROL atraves del tiempo')
plt.xlabel('Año')
plt.ylabel('Precio (COP)')


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


















