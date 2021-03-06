import pandas as pd
import matplotlib.pyplot as plt
import datetime
import unidecode
import re
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

noticias_esp_eco = pd.read_excel('Araña  larepublica ecopetrol.xlsx')
noticias_esp_ban = pd.read_excel('Araña  larepublica bancolombia.xlsx')
noticias_esp_col = pd.read_excel('Araña  larepublica colcap.xlsx')


def preprocesamiento(noticias_esp):
    if len(noticias_esp) == len(noticias_esp_eco):
        sheet_val = 'english_eco'
    elif len(noticias_esp) == len(noticias_esp_ban):
        sheet_val = 'english_banco'
    else:
        sheet_val = 'english_colcap'

    noticias_esp = noticias_esp.dropna()  # Eliminar filas vacias

    noticias_esp = noticias_esp.drop_duplicates()  # Eliminar duplicados
    noticias_esp['Noticia'] = noticias_esp['Title']
    noticias_esp = noticias_esp.drop(['Text'], 1)

    # noticias_esp['Noticia'] = noticias_esp['Title'] + '. ' + noticias_esp[
    #     'Text']  # concatenar las columnas Title y Text dentro de una sola, Noticia.
    #
    # noticias_esp = noticias_esp.drop(['Title', 'Text'], 1)  # Descartar las columnas sobrantes

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

    noticias_esp = noticias_esp.reset_index(drop=True)

    fechas = pd.Series(map(ajustar_fecha, noticias_esp['Fecha']))  # Convertir el iterable map a una serie de pandas
    noticias_esp['Fecha'] = fechas  # llevar al DataFrame los datos de fecha modificados
    noticias_esp = noticias_esp.set_index('Fecha')
    # Editando la columna Noticia

    # Limpieza con el texto traducido al ingles

    "t.to_excel('noticias larepublica.xlsx', encoding='utf-8')"

    def limpieza_para_traducir(texto):
        texto = re.sub(r'ARTÍCULO RELACIONADO\n[\w\s,%\-"?¿¡!:$.&“”]+\n+', '',
                       texto)  # Remover texto de articulo relacionado
        texto = re.sub(r'\s+', ' ', texto)

        return texto

    # noticias_esp_fix = noticias_esp.copy()
    #
    # noticias_esp_fix['Noticia'] = [limpieza_para_traducir(texto) for texto in noticias_esp.Noticia]
    # noticias_esp_fix.to_excel('noticias larepublica_fixed.xlsx', encoding='utf-8')

    noticias_ing = pd.read_excel("noticias larepublica.xlsx", sheet_name=sheet_val, header=None)
    noticias_ing.columns = ['Noticia']

    noticias_ing['Fecha'] = noticias_esp.index
    noticias_ing = noticias_ing.set_index('Fecha')

    def limpieza_general(texto):
        texto = re.sub(r'ARTÍCULO RELACIONADO\n[\w\s,%\-"?¿¡!:.]+\n+', '',
                       texto)  # Remover texto de articulo relacionado

        texto = texto.lower()  # Establecer texto como minuscula

        texto = re.sub(r'\s+', ' ', texto)  # Quitar espacios y entre lineas

        texto = re.sub('[^\w\s]', '', texto)  # Eliminar simbolos y puntuaciones

        texto = re.sub(r'\w*\d\w*', '', texto)  # Eliminnar numeros y cadenas con numeros

        texto = unidecode.unidecode(texto)  # Eliminar diacriticos y ñs

        return texto

    noticias_ing_pp = noticias_ing.copy()

    noticias_ing_pp['Noticia'] = [limpieza_general(noticia) for noticia in noticias_ing.Noticia]

    noticias_esp_pp = noticias_esp.copy()

    noticias_esp_pp['Noticia'] = [limpieza_general(noticia) for noticia in noticias_esp.Noticia]

    return noticias_ing_pp


noticias_ecopetrol_ing = preprocesamiento(noticias_esp_eco)
noticias_bancolombia_ing = preprocesamiento(noticias_esp_ban)
notcias_colcap_ing = preprocesamiento(noticias_esp_col)

# EDA matplotlib


# Wordcloud

corpus_list = [noticia for noticia in notcias_colcap_ing['Noticia']]
corpus_str = ' '.join(corpus_list)
text_cloud = WordCloud(width=2000, height=1000, background_color='white',
                       stopwords=stopwords.words('spanish')).generate(corpus_str)
plt.figure(figsize=(20, 10))
text_cloud.to_file('wordcloud1.png')
plt.axis("off")
plt.imshow(text_cloud, interpolation='bilinear')
plt.show()

ecopetrol_pe = noticias_ecopetrol_ing.loc['2016-03-01':'2016-04-07']

# Textblob

from textblob import TextBlob

# Texto completo

blob = [TextBlob(text) for text in ecopetrol_pe['Noticia']]
ecopetrol_pe['Textblob'] = [round(blob[i].sentiment.polarity, 3) for i in range(len(blob))]
text = "Ecopetrol's reserves decreased 11% and would last for 7.4 years. Now, despite the fact that this was the result that the stock market analysts were waiting for, as BTG Pactual anticipated, “so no large movements are expected in the stock since the market has anticipated most of this and the Ecopetrol's performance will be more linked to oil prices at this time ”, it is worrying that the reserve replacement ratio was only 6%, while in previous years it was 146% and 139% in 2013. This is how only 67 Mboe correspond to new drilling campaigns and positive reviews in some fields, such as Chichimene. On the other hand, in the face of new discoveries, only 24 Mboe were incorporated in the previous year, that is, 52% less than in 2014, when 50 mbpe were added, due to the incorporation of some gas fields, which helped that in 2014 it was possible to establish that the reserves would last for 8.5 years. Finally, BTG pointed out that the fall in reserves is likely to represent an impediment to spending that could erase the 2015 earnings. More so if one takes into account that Total, ENI, Chevron and Statoil were above the oil company, which decreased their figures in a lower percentage."
del(ecopetrol_pe['Textblob (p,s)'])

# Solamente titulo


# VADER

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

text = 'ecopetrol lost the first place as the most valuable company in latin america the colombian oil company was displaced from the top position by the brazilian brewer ambev in the list by market value of latin american companies in  according to a study by the economatica consultancy the value of ambev according to the consulting company was us   billion followed by ecopetrol with us   billion the third position was occupied by the brazilian oil company petrobras with us   billion the brazilian mining company vale was in fourth place with us   million according to the report that was made based on the price of the shares of the companies in the different exchanges that are listed america movil from mexico was the fifth most valuable company in latin america it is followed by itauunibanco and bradesco wal mart de mexico banco de brasil and femsa'

text_tokenized = word_tokenize(text)
stopwords = stopwords.words('english')
text_filtered = [word for word in text_tokenized if word not in stopwords]
text_filtered = ' '.join(text_filtered)
new_sentiment = SentimentIntensityAnalyzer()

ecopetrol_pe['Vader compound'] = [new_sentiment.polarity_scores(new)['compound'] for new in
                                     ecopetrol_pe['Noticia']]

# SenticNet 6.0


from nltk import word_tokenize
from nltk import bigrams
import senticnet6_polarity
from senticnet.senticnet6 import senticnet

sentiment_words = {}


def polarity(list_of_words, dictionary):
    pos = 0
    neg = 0
    poslist = []
    neglist = []

    for word in list_of_words:
        try:
            if float(dictionary[word][7]) > 0:
                pos += 1
                poslist.append(word)
                if word not in sentiment_words:
                    sentiment_words.update({word: 1})
                else:
                    sentiment_words.update({word: sentiment_words[word] + 1})
            elif float(dictionary[word][7]) < 0:
                neg += 1
                neglist.append(word)
                if word not in sentiment_words:
                    sentiment_words.update({word: 1})
                else:
                    sentiment_words.update({word: sentiment_words[word] + 1})
        except  KeyError:
            continue
    if pos + neg != 0:
        pol = (pos - neg) / (pos + neg)
    else:
        pol = 0

    return [pol, [pos, neg], [poslist, neglist]]


# def polarity_polarity(list_of_words, dictionary):
#     pos = 0
#     neg = 0
#     poslist = []
#     neglist = []
#
#     for word in list_of_words:
#         try:
#             if float(dictionary[word][7]) > 0:
#                 pos += float(dictionary[word][7])
#                 poslist.append(word)
#
#             elif float(dictionary[word][7]) < 0:
#                 neg += float(dictionary[word][7])
#                 neglist.append(word)
#
#
#         except  KeyError:
#             continue
#
#     return [[pos + neg], [poslist, neglist]]
#

def bigrams_(text):
    tokens = word_tokenize(text)
    bi_grams = ['_'.join(i) for i in list(bigrams(tokens))]
    return bi_grams


# noticias_ing_pp_t_bg = noticias_ing_pp.copy()
# noticias_ing_pp_t_bg['Noticia'] = [bigrams_(new) for new in noticias_ing_pp_t_bg.Noticia]
# noticias_ing_pp_t_bg['polarity_label'] = [polarity(new, senti
# cnet) for new in noticias_ing_pp_t_bg.Noticia]
# noticias_ing_pp_t_bg['polarity_value'] = [polarity_polarity(new, senticnet) for new in noticias_ing_pp_t_bg.Noticia]

ecopetrol_pe_t = ecopetrol_pe.copy()
stopwords = ['oil', 'production', 'crude', 'sector', 'energy', 'gas', 'pipeline', 'refinery', 'petroleum', 'dollar',
             'infrastructure', 'water', 'hydrocarbon', 'latin', 'stocks', 'gasoline', 'exploitation']
ecopetrol_pe_t['Noticia'] = [new.split() for new in ecopetrol_pe_t.Noticia]
ecopetrol_pe_t['Noticia'] = ecopetrol_pe_t['Noticia'].apply(
    lambda x: [word for word in x if word not in stopwords])
ecopetrol_pe['Senticnet polarity'] = [polarity(new, senticnet)[0] for new in ecopetrol_pe_t.Noticia]

a = dict(sorted(sentiment_words.items(), key=lambda x: x[1], reverse=True))

# noticias_ing_pp_t['polarity_value'] = [polarity_polarity(new, senticnet) for new in noticias_ing_pp_t.Noticia]

# sample0 = [[0, 0], [[], []]]
#
# sample2 = [[1, 0], [['merger_agreement'], []]]
# sample1 = [[14, 4], [['oil', 'announced', 'decision', 'merge', 'board', 'decision', 'board', 'agreement', 'registered', 'absorbing', 'registration', 'registrar', 'registrar', 'competent'], ['superintendency', 'caiman', 'cayman', 'cayman']]]
#
#
# def merge(text1, text2):
#
#    if text2[0][0]  != 0:
#        text3 = [term.split('_')for term in text2[1][0]]


# Loughram and McDonald

lm_dic_neg = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name='Negative', header=None)
lm_dic_pos = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name='Positive', header=None)


def lm(dic_pos, dic_neg):
    lm_dic = {str(key).lower(): -1 for key in dic_neg}
    lm_dic.update({str(key).lower(): 1 for key in dic_pos})

    return lm_dic


def lm_polarity(new, lm_dic):
    pos = 0
    neg = 0
    poslist = []
    neglist = []

    for token in new:
        try:
            if lm_dic[token] > 0:
                pos += 1
                poslist.append(token)
            elif lm_dic[token] < 0:
                neg += 1
                neglist.append(token)

        except  KeyError:
            continue

    if pos + neg != 0:
        pol = (pos - neg) / (pos + neg)
    else:
        pol = 0

    return [pol, [pos, neg], [poslist, neglist]]


lm_dic = lm(lm_dic_pos.iloc[:, 0], lm_dic_neg.iloc[:, 0])

ecopetrol_pe['LM polarity'] = [lm_polarity(new, lm_dic)[0] for new in ecopetrol_pe_t['Noticia']]

eco