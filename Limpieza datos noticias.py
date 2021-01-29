import pandas as pd
import datetime
import unidecode
import re
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


noticias_esp = pd.read_excel('larepublica raw data.xlsx')

noticias_esp = noticias_esp.dropna()  # Eliminar filas vacias
noticias_esp = noticias_esp.drop_duplicates()  # Eliminar duplicados

noticias_esp['Noticia'] = noticias_esp['Title'] + '. ' + noticias_esp[
    'Text']  # concatenar las columnas Title y Text dentro de una sola, Noticia.

noticias_esp = noticias_esp.drop(['Title', 'Text'], 1)  # Descartar las columnas sobrantes

noticias_esp.rename(columns={'Pub_Time': 'Fecha'}, inplace=True)  # Cambiar nombre de la columna fecha


def ajustar_fecha(date):
    """Función para transformar la fecha al formato estandar .

    Args:
        date (str): Fecha no estructurada ej: jueves, 3 de enero de 2013

    Returns:
        fecha_corregida (str): Fecha corregida en formato estandar ej: 2013-01-03

"""

    dic = {'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'mayo': '05', 'junio': '06', 'julio': '07',
           'agosto': '08', 'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'}  # Diccionario
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

    fecha_corregida = '-'.join(fecha_corregida)  # Se convierte la variable de lista a string, con separador "-" ej:
    # ['2013', '01', '15'] -> '2013-01-15

    return fecha_corregida


fechas = pd.Series(map(ajustar_fecha, noticias_esp['Fecha']))  # Convertir el iterable map a una serie de pandas
noticias_esp['Fecha'] = fechas  # llevar al DataFrame los datos de fecha modificados

# Editando la columna Noticia

# Limpieza con el texto traducido al ingles
test_text = noticias_esp['Noticia']


def eliminar_espacios(texto):
    pattern = re.compile(r'\s+')
    texto_sin_espacios = re.sub(pattern, ' ', texto)
    return texto_sin_espacios


t = pd.Series(map(eliminar_espacios, test_text))

"t.to_excel('noticias larepublica.xlsx', encoding='utf-8')"

noticias_columna_ing = pd.read_excel("noticias larepublica.xlsx", sheet_name='english', header=None)

noticias_ing = pd.DataFrame(noticias_esp['Fecha'])
noticias_ing['Noticias'] = noticias_columna_ing


def limpieza_general(texto):
    texto = texto.lower()  # Establecer todo como minuscula

    texto = re.sub(r'\s+', ' ', texto)  # Quitar espacios y entre lineas

    texto = re.sub('[^\w\s]', '', texto)  # Eliminar simbolos y puntuaciones

    texto = re.sub(r'\w*\d\w*', '', texto)  # Eliminnar numeros y cadenas con numeros

    texto = unidecode.unidecode(texto)  # Eliminar diacriticos y ñs

def tokenizacion_unigrams(texto):


def tokenizacion_bigrams(texto):

