import pandas as pd
import datetime
import re


noticias_republica = pd.read_excel('larepublica raw data.xlsx')


noticias_republica = noticias_republica.dropna()  # Eliminar filas vacias
noticias_republica = noticias_republica.drop_duplicates()  # Eliminar duplicados

noticias_republica['Noticia'] = noticias_republica['Title'] + '. ' + noticias_republica[
    'Text']  # concatenar las columnas Title y Text dentro de una sola, Noticia.


noticias_republica = noticias_republica.drop(['Title', 'Text'], 1)  # Descartar las columnas sobrantes


noticias_republica.rename(columns={'Pub_Time': 'Fecha'}, inplace=True)  # Cambiar nombre de la columna fecha


def ajustar_fecha(date):
    dic = {'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'mayo': '05', 'junio': '06', 'julio': '07',
           'agosto': '08', 'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'}

    fecha_corregida = []

    fecha = date.split()

    del fecha[0]
    del fecha[1]
    del fecha[2]

    fecha_corregida.append(fecha[2])

    if fecha[1] in dic.keys():
        fecha_corregida.append(dic[fecha[1]])

    if len(fecha[0]) == 1:
        fecha_corregida.append('0' + fecha[0])
    else:
        fecha_corregida.append(fecha[0])

    fecha_corregida = '-'.join(fecha_corregida)

    return fecha_corregida


fechas = pd.Series(map(ajustar_fecha, noticias_republica['Fecha']))
noticias_republica['Fecha'] = fechas


# Editando la columna Noticia



