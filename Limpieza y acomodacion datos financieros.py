
# Ajuste datos financieros


import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)  # Setup para ampliar visualización de datos (en este caso DataFrame) en el IDE
pd.set_option('display.width', 1000)


ecopetrol_df = pd.read_excel("Ecopetrol 2013-2019 raw data.xls")  # Importar datos financieros como Dataframe, (previamente convertidos de
# Formato xls a csv) usando pandas



keep_columns = ['fecha', 'Volumen', 'Precio Cierre', 'Precio Mayor', 'Precio Menor', 'Variacion Absoluta']   # Se
# Establecen las columnas que se van a usar,"Cantidad de transacciones" y "Precio Medio" se descartan


ecopetrol_df['Precio Apertura'] = ecopetrol_df['Precio Cierre'] + ecopetrol_df['Variacion Absoluta']
# Se calcula el valor "Open" de la accion



columns_fixed = ['fecha', 'Precio Apertura', 'Precio Mayor', 'Precio Menor', 'Precio Cierre', 'Volumen']
ecopetrol_df = ecopetrol_df[columns_fixed]
# Orden de las columnas de datos, según formato requerido




# Establecer la fecha como indice para que el indice predeterminado (0,1,2,3,4..) no se agregue como una columna nueva

ecopetrol_df = ecopetrol_df.set_index('fecha')

print(ecopetrol_df)


ecopetrol_csv_cleaned = ecopetrol_df.to_csv('Ecopetrol limpio.csv', encoding='utf-8')
# Conversión y guardado de los datos en disco
ecopetrol_csv_limpio = pd.read_csv('Ecopetrol limpio.csv')

print(ecopetrol_csv_limpio)