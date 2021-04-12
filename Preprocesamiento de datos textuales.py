import pandas as pd
import datetime
import random as rd
import unidecode
import re
import pickle
empresas = ['ecopetrol', 'bancolombia', 'colcap']
dic_titles = {}
noticias_writer_titulos = pd.ExcelWriter('noticias larepublica_title.xlsx', engine='xlsxwriter')
noticias_writer_completo = pd.ExcelWriter('noticias larepublica_completo.xlsx', engine='xlsxwriter')
for i in range(len(empresas)):

    noticias = pd.read_excel('araña larepublica ' + str(empresas[i]) + '.xlsx')
    noticias = noticias.drop(axis=1, columns=['Text'])
    noticias.columns = ['noticia', 'fecha']


    def ajustar_fecha(date):
        """Función para transformar la fecha al formato estandar .

        Args:
            date (str): Fecha no estructurada ej: jueves, 3 de enero de 2013

        Returns:
            fecha_corregida (str): Fecha corregida en formato estandar ej: 2013-01-03

    """

        dic = {'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'mayo': '05', 'junio': '06',
               'julio': '07',
               'agosto': '08', 'septiembre': '09', 'octubre': '10', 'noviembre': '11',
               'diciembre': '12'}  # Diccionario
        # para convertir el mes de texto a numero, en formato str

        fecha_corregida = []  # Output en lista

        fecha = date.split()  # Dividir con el delimitador " " cada palabra que conforma la celda Fecha, formando un objeto lista

        del fecha[0]  # Depurar día de la semana "lunes..."
        del fecha[1]  # Depurar primera preposicion "de"
        del fecha[2]  # Depurar segunda preposicion "de"

        fecha_corregida.append(fecha[2])  # Añade el año a la variable de salida

        if fecha[1] in dic.keys():
            fecha_corregida.append(dic[fecha[1]])  # Si el segundo elemento de la variable de entrada (convertida a
            # lista), esta dentro de las llaves del diccionario dic. Añade el mes a la variable de salida

        if len(fecha[0]) == 1:
            fecha_corregida.append('0' + fecha[0])  # Se añade un 0 al dia del mes, ej 2013-01-5 -> 2013-01-05
        else:
            fecha_corregida.append(fecha[0])  # Si el día tiene 2 digitos, no se altera

        fecha_corregida = '-'.join(
            fecha_corregida)  # Se convierte la variable de lista a string, con separador "-" ej:
        # ['2013', '01', '15'] -> '2013-01-15
        fecha_corregida = datetime.datetime.strptime(fecha_corregida, '%Y-%m-%d')

        return fecha_corregida


    noticias = noticias.dropna()
    noticias = noticias.drop_duplicates()
    noticias['fecha'] = noticias['fecha'].apply(ajustar_fecha)
    noticias = noticias.set_index('fecha')


    def limpieza_general(texto):
        texto = re.sub(r'ARTÍCULO RELACIONADO\n[\w\s,%\-"?¿¡!:.]+\n+', '',
                       texto)  # Remover texto de articulo relacionado

        texto = texto.lower()  # Establecer texto como minuscula

        texto = re.sub(r'\s+', ' ', texto)  # Quitar espacios y entre lineas

        texto = re.sub('[^\w\s]', '', texto)  # Eliminar simbolos y puntuaciones

        texto = re.sub(r'\w*\d\w*', '', texto)  # Eliminnar numeros y cadenas con numeros

        texto = unidecode.unidecode(texto)  # Eliminar diacriticos y ñs

        return texto


    noticias['noticia'] = noticias['noticia'].apply(limpieza_general)
    s_noticias = pd.Series(data=noticias.iloc[:, 0], index=noticias.index)
    dic_titles['traducir ' + str(empresas[i])] = s_noticias

    dic_titles['traducir ' + str(empresas[i])].to_excel(noticias_writer_titulos, sheet_name=empresas[i])
noticias_writer_titulos.save()
noticias_writer_completo.close()

for i in range(len(empresas)):

    noticias = pd.read_excel('araña larepublica ' + str(empresas[i]) + '.xlsx')
    noticias['noticia'] = noticias['Title'] + '. ' + noticias['Text']
    noticias = noticias.drop(columns=['Title', 'Text'])
    noticias.columns = ['fecha', 'noticia']


    def ajustar_fecha(date):
        """Función para transformar la fecha al formato estandar .

        Args:
            date (str): Fecha no estructurada ej: jueves, 3 de enero de 2013

        Returns:
            fecha_corregida (str): Fecha corregida en formato estandar ej: 2013-01-03

    """

        dic = {'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'mayo': '05', 'junio': '06',
               'julio': '07',
               'agosto': '08', 'septiembre': '09', 'octubre': '10', 'noviembre': '11',
               'diciembre': '12'}  # Diccionario
        # para convertir el mes de texto a numero, en formato str

        fecha_corregida = []  # Output en lista

        fecha = date.split()  # Dividir con el delimitador " " cada palabra que conforma la celda Fecha, formando un objeto lista

        del fecha[0]  # Depurar día de la semana "lunes..."
        del fecha[1]  # Depurar primera preposicion "de"
        del fecha[2]  # Depurar segunda preposicion "de"

        fecha_corregida.append(fecha[2])  # Añade el año a la variable de salida

        if fecha[1] in dic.keys():
            fecha_corregida.append(dic[fecha[1]])  # Si el segundo elemento de la variable de entrada (convertida a
            # lista), esta dentro de las llaves del diccionario dic. Añade el mes a la variable de salida

        if len(fecha[0]) == 1:
            fecha_corregida.append('0' + fecha[0])  # Se añade un 0 al dia del mes, ej 2013-01-5 -> 2013-01-05
        else:
            fecha_corregida.append(fecha[0])  # Si el día tiene 2 digitos, no se altera

        fecha_corregida = '-'.join(
            fecha_corregida)  # Se convierte la variable de lista a string, con separador "-" ej:
        # ['2013', '01', '15'] -> '2013-01-15
        fecha_corregida = datetime.datetime.strptime(fecha_corregida, '%Y-%m-%d')

        return fecha_corregida


    noticias = noticias.dropna()
    noticias = noticias.drop_duplicates()
    noticias['fecha'] = noticias['fecha'].apply(ajustar_fecha)
    noticias = noticias.set_index('fecha')


    def limpieza_para_traducir(texto):
        texto = re.sub(r'ARTÍCULO RELACIONADO\n[\w\s,%\-"?¿¡!:$.&“”]+\n+', '',
                       texto)  # Remover texto de articulo relacionado
        texto = re.sub(r'\s+', ' ', texto)

        return texto


    noticias['noticia'] = noticias['noticia'].apply(limpieza_para_traducir)
    s_noticias = pd.Series(data=noticias.iloc[:, 0], index=noticias.index)
    dic_news['traducir ' + str(empresas[i])] = s_noticias

    dic_news['traducir ' + str(empresas[i])].to_excel(noticias_writer_completo, sheet_name=empresas[i])

noticias_writer_completo.save()
noticias_writer_completo.close()