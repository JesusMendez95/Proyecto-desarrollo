import pandas as pd
import matplotlib.pyplot as plt

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

noticias_esp_eco = pd.read_excel('araña larepublica ecopetrol.xlsx')
noticias_esp_ban = pd.read_excel('araña larepublica bancolombia.xlsx')
noticias_esp_col = pd.read_excel('araña larepublica colcap.xlsx')

# def preprocesamiento(noticias_esp):
#     if len(noticias_esp) == len(noticias_esp_eco):
#         sheet_val = 'english_eco'
#     elif len(noticias_esp) == len(noticias_esp_ban):
#         sheet_val = 'english_banco'
#     else:
#         sheet_val = 'english_colcap'
# #
#     noticias_esp = noticias_esp.dropna()  # Eliminar filas vacias
#
#     noticias_esp = noticias_esp.drop_duplicates()  # Eliminar duplicados
#
#     noticias_esp['Noticia'] = noticias_esp['Title'] + '. ' + noticias_esp[
#         'Text']  # concatenar las columnas Title y Text dentro de una sola, Noticia.
#
#     noticias_esp = noticias_esp.drop(['Title', 'Text'], 1)  # Descartar las columnas sobrantes
#
#     noticias_esp.rename(columns={'Pub_Time': 'Fecha'}, inplace=True)  # Cambiar nombre de la columna fecha
#
# def ajustar_fecha(date):
#     """Función para transformar la fecha al formato estandar .
#
#     Args:
#         date (str): Fecha no estructurada ej: jueves, 3 de enero de 2013
#
#     Returns:
#         fecha_corregida (str): Fecha corregida en formato estandar ej: 2013-01-03
#
# """
#
#     dic = {'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'mayo': '05', 'junio': '06', 'julio': '07',
#            'agosto': '08', 'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'}  # Diccionario
#     # para convertir el mes de texto a numero, en formato str
#
#     fecha_corregida = []  # Output en lista
#
#     fecha = date.split()  # Dividir con el delimitador " " cada palabra que conforma la celda Fecha, formando un objeto lista
#
#     del fecha[0]  # Depurar día de la semana "lunes..."
#     del fecha[1]  # Depurar primera preposicion "de"
#     del fecha[2]  # Depurar segunda preposicion "de"
#
#     fecha_corregida.append(fecha[2])  # Añade el año a la variable de salida
#
#     if fecha[1] in dic.keys():
#         fecha_corregida.append(dic[fecha[1]])  # Si el segundo elemento de la variable de entrada (convertida a
#         # lista), esta dentro de las llaves del diccionario dic. Añade el mes a la variable de salida
#
#     if len(fecha[0]) == 1:
#         fecha_corregida.append('0' + fecha[0])  # Se añade un 0 al dia del mes, ej 2013-01-5 -> 2013-01-05
#     else:
#         fecha_corregida.append(fecha[0])  # Si el día tiene 2 digitos, no se altera
#
#     fecha_corregida = '-'.join(fecha_corregida)  # Se convierte la variable de lista a string, con separador "-" ej:
#     # ['2013', '01', '15'] -> '2013-01-15
#
#     return fecha_corregida
#
#     noticias_esp = noticias_esp.reset_index(drop=True)
#
#     fechas = pd.Series(map(ajustar_fecha, noticias_esp['Fecha']))  # Convertir el iterable map a una serie de pandas
#     noticias_esp['Fecha'] = fechas  # llevar al DataFrame los datos de fecha modificados
#     noticias_esp = noticias_esp.set_index('Fecha')
#
#     # # Editando la columna Noticia
#     #
#     # # Limpieza con el texto traducido al ingles
#     #
#     # # noticias_esp.to_excel('noticias larepublica_fixed.xlsx', encoding='utf-8')
#     #
#     def limpieza_para_traducir(texto):
#         texto = re.sub(r'ARTÍCULO RELACIONADO\n[\w\s,%\-"?¿¡!:$.&“”]+\n+', '',
#                        texto)  # Remover texto de articulo relacionado
#         texto = re.sub(r'\s+', ' ', texto)
#
#         return texto
# #
#
#     def limpieza_general(texto):
#         texto = re.sub(r'ARTÍCULO RELACIONADO\n[\w\s,%\-"?¿¡!:.]+\n+', '',
#                        texto)  # Remover texto de articulo relacionado
#
#         texto = texto.lower()  # Establecer texto como minuscula
#
#         texto = re.sub(r'\s+', ' ', texto)  # Quitar espacios y entre lineas
#
#         texto = re.sub('[^\w\s]', '', texto)  # Eliminar simbolos y puntuaciones
#
#         texto = re.sub(r'\w*\d\w*', '', texto)  # Eliminnar numeros y cadenas con numeros
#
#         texto = unidecode.unidecode(texto)  # Eliminar diacriticos y ñs
#
#         return texto


#
# # EDA matplotlib
#
#
# # Wordcloud
#
# corpus_list = [noticia for noticia in notcias_colcap_ing['Noticia']]
# corpus_str = ' '.join(corpus_list)
# text_cloud = WordCloud(width=2000, height=1000, background_color='white',
#                        stopwords=stopwords.words('spanish')).generate(corpus_str)
# plt.figure(figsize=(20, 10))
# text_cloud.to_file('wordcloud1.png')
# plt.axis("off")
# plt.imshow(text_cloud, interpolation='bilinear')
# plt.show()


# Textblob

from textblob import TextBlob
from textblob.tokenizers import SentenceTokenizer

# Texto completo


df = pd.read_excel('noticias larepublica.xlsx', sheet_name='english_eco', index_col=0)

df.columns = ['noticia']
df.noticia = [string.replace(u'\xa0', u' ') for string in df['noticia']]
df_sin_pp = df.copy()

sentence_tokenizer = SentenceTokenizer()
df.noticia = [sentence_tokenizer.tokenize(text) for text in df.noticia]

blob = [TextBlob(paragraphs) for new in df.noticia for paragraphs in new]
blob2 = [blob[i].sentiment.polarity for i in range(len(blob))]

# ecopetrol_pe['Textblob'] = [round(blob[i].sentiment.polarity, 3) for i in range(len(blob))]

tb = [[TextBlob(i) for i in paragr] for paragr in df.noticia]
df_copy = df.noticia.copy()
df_copy.iloc[:] = tb
sentiments = [[row[i].sentiment.polarity for i in range(len(row))] for row in tb]
df_copy.iloc[:] = sentiments


def blob_mod(list_p):
    blob2_mod = []
    for par in list_p:
        if par < 0:
            blob2_mod.append(par)
        elif 0 <= par <= 0.25:
            pass
        else:
            blob2_mod.append(par)
    if len(blob2_mod) != 0:
        blob2_mod = sum(blob2_mod) / len(blob2_mod)
    else:
        blob2_mod = 0

    return blob2_mod


eco_sentiment_sentece = [blob_mod(row) for row in df_copy]
df_copy.iloc[:] = eco_sentiment_sentece
new_ss_textblob = df_copy.copy()
new_ss_textblob.iloc[:] = eco_sentiment_sentece

# blob = [TextBlob(text) for text in ecopetrol_pe['Noticia']]
# ecopetrol_pe['Textblob'] = [round(blob[i].sentiment.polarity, 3) for i in range(len(blob))]


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

vader = [[new_sentiment.polarity_scores(parragraph)['compound'] for parragraph in row] for row in
         df.noticia]


def vader_modify(list_p):
    vader_mod = []
    for par in list_p:
        if par < 0:
            vader_mod.append(par)
        elif 0 <= par <= 0.5:
            pass
        else:
            vader_mod.append(par)
    if len(vader_mod) != 0:
        vader_mod = sum(vader_mod) / len(vader_mod)
    else:
        vader_mod = 0

    return vader_mod


vader_eco_ss = [vader_modify(row) for row in vader]
vader_eco_ss_d = df_copy.copy()
vader_eco_ss_d.iloc[:] = vader_eco_ss

# ecopetrol_pe['Vader compound'] = [new_sentiment.polarity_scores(new)['compound'] for new in
#                                      ecopetrol_pe['Noticia']]

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

# def bigrams_(text):
#     tokens = word_tokenize(text)
#     bi_grams = ['_'.join(i) for i in list(bigrams(tokens))]
#     return bi_grams


# noticias_ing_pp_t_bg = noticias_ing_pp.copy()
# noticias_ing_pp_t_bg['Noticia'] = [bigrams_(new) for new in noticias_ing_pp_t_bg.Noticia]
# noticias_ing_pp_t_bg['polarity_label'] = [polarity(new, senti
# cnet) for new in noticias_ing_pp_t_bg.Noticia]
# noticias_ing_pp_t_bg['polarity_value'] = [polarity_polarity(new, senticnet) for new in noticias_ing_pp_t_bg.Noticia]
# def limpieza_general(texto):
#     texto = re.sub(r'ARTÍCULO RELACIONADO\n[\w\s,%\-"?¿¡!:.]+\n+', '',
#                    texto)  # Remover texto de articulo relacionado
#
#     texto = texto.lower()  # Establecer texto como minuscula
#
#     texto = re.sub(r'\s+', ' ', texto)  # Quitar espacios y entre lineas
#
#     texto = re.sub('[^\w\s]', '', texto)  # Eliminar simbolos y puntuaciones
#
#     texto = re.sub(r'\w*\d\w*', '', texto)  # Eliminnar numeros y cadenas con numeros
#
#     texto = unidecode.unidecode(texto)  # Eliminar diacriticos y ñs
#
#     return texto


df_empresa = df_sin_pp.copy()
df_empresa_tokenized = df_empresa.copy()
df_empresa_tokenized.noticia = [limpieza_general(noticia) for noticia in df_sin_pp.noticia]

df_empresa_tokenized.noticia = [new.split() for new in df_empresa_tokenized.noticia]
stopwords = ['oil', 'production', 'crude', 'sector', 'energy', 'gas', 'pipeline', 'refinery', 'petroleum', 'dollar',
             'infrastructure', 'water', 'hydrocarbon', 'latin', 'stocks', 'gasoline', 'exploitation']
df_empresa_tokenized.noticia = df_empresa_tokenized.noticia.apply(
    lambda x: [word for word in x if word not in stopwords])
senticnet_pol = df_empresa_tokenized.copy()
senticnet_pol.noticia = [polarity(new, senticnet)[0] for new in df_empresa_tokenized.noticia]

# a = dict(sorted(sentiment_words.items(), key=lambda x: x[1], reverse=True))

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
lm_pol = df_empresa_tokenized.copy()
lm_pol.noticia = [lm_polarity(new, lm_dic)[0] for new in lm_pol.noticia]

# # Metricas prueba de escritorio
# pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_columns',
#               1000)  # Setup para ampliar visualización de datos (en este caso DataFrame) en el IDE
# pd.set_option('display.width', 1000)
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report
# import seaborn as sns
#
# df_pe_inicial = pd.read_excel('prueba escritorio python.xlsx', sheet_name='Prueba escritorio completa')
#
# df_pe_titulos = pd.read_excel('prueba escritorio python.xlsx', sheet_name='Prueba escritorio titulos')
# labels = ["Positivo", "Neutro", "Negativo"]
# y_true_inicial = df_pe_titulos.iloc[:, [0, 5, 10]]
# y_pred_inicial = df_pe_titulos.iloc[:, [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]]
# test = classification_report(df_pe_titulos.iloc[:, 5], df_pe_titulos.iloc[:, 9], labels=labels, zero_division=True)
# l = []
# m = y_true_inicial.iloc[:,0].value_counts() + y_true_inicial.iloc[:,1].value_counts() + y_true_inicial.iloc[:,2].value_counts()
#
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 10))
#
# # cm_df = eco_df.iloc[:, [1, 2, 3, 4]]
#
# # cms = [confusion_matrix(y_pred_inicial.iloc[:, ], y_pred_inicial[:, i]) for i in enumerate(12)]
# cms = []
# i = 0
# for j in range(len(y_pred_inicial.columns)):
#     if j == 4 or j == 8:
#         i += 1
#     cms.append(confusion_matrix(y_true_inicial.iloc[:, i], y_pred_inicial.iloc[:, j],labels=labels))
#
# crs = []
# i = 0
# for j in range(len(y_pred_inicial.columns)):
#     if j == 4 or j == 8:
#         i += 1
#     crs.append(classification_report(y_true_inicial.iloc[:, i], y_pred_inicial.iloc[:, j],labels=labels, output_dict=True, zero_division=True))
#
# weighted_precision = [element['precision']for element in crs]
# weighted_recall = [element['recall'] for element in crs]
# weighted_f1_score = [element['f1-score'] for element in crs]
#
# def weighted_by_classifier(wp, wr, wf1):
#     weighted_textblob = {'precision': sum([wp[0], wp[4], wp[8]]) / 3, 'recall': sum([wr[0], wr[4], wr[8]]) / 3,
#                          'f1-score': sum([wf1[0], wf1[4], wf1[8]]) / 3}
#
#     weighted_vader = {'precision': sum([wp[1], wp[5], wp[9]]) / 3, 'recall': sum([wr[1], wr[5], wr[9]]) / 3,
#                       'f1-score': sum([wf1[1], wf1[5], wf1[9]]) / 3}
#     weighted_senticnet = {'precision': sum([wp[2], wp[6], wp[10]]) / 3, 'recall': sum([wr[2], wr[6], wr[10]]) / 3,
#                       'f1-score': sum([wf1[2], wf1[6], wf1[10]]) / 3}
#     weighted_LM = {'precision': sum([wp[3], wp[7], wp[11]]) / 3, 'recall': sum([wr[3], wr[7], wr[11]]) / 3,
#                           'f1-score': sum([wf1[3], wf1[7], wf1[11]]) / 3}
#
#     return weighted_textblob, weighted_vader, weighted_senticnet, weighted_LM
#
# metrics = weighted_by_classifier(weighted_precision,weighted_recall,weighted_f1_score)
# metrics_df = pd.DataFrame(metrics)
# writer_metrics = pd.ExcelWriter('enfoque titulares metricas .xlsx', engine="xlsxwriter")
# metrics_df.to_excel(writer_metrics, sheet_name='reporte de clasificacion')
#
# i = 0
# np_tl = np.zeros((9, 1))
#
# df = pd.DataFrame(index=[
#     ['Prediccion Perfecta', 'Prediccion Perfecta', 'Prediccion Perfecta', 'Prediccion Mediocre',
#      'Prediccion Mediocre', 'Prediccion Mediocre', 'Prediccion Mediocre', 'Predicción Incorrecta',
#      'Predicción Incorrecta'],
#     ['positiva-positiva', 'neutra-netra', 'negativa-negativa', 'positiva-neutra', 'negativa-neutra',
#      'neutra-positiva', 'neutra-negativa', 'positiva-negativa', 'negativa-positiva']],
#     columns=['Textblob_eco', 'Vader_eco', 'SenticNet_eco', 'LM_eco', 'Textblob_b', 'Vader_b',
#              'SenticNet_b', 'LM_b', 'Textblob_c', 'Vader_c', 'SenticNet_c', 'LM_c'],
#     data=np.zeros(108).reshape((9, 12)))
#
# dummy = []
# for cm in cms:
#     dummy.append(cm.flat[0])
#     dummy.append(cm.flat[4])
#     dummy.append(cm.flat[8])
#     dummy.append(cm.flat[1])
#     dummy.append(cm.flat[3])
#     dummy.append(cm.flat[5])
#     dummy.append(cm.flat[7])
#     dummy.append(cm.flat[2])
#     dummy.append(cm.flat[6])
#
#     df.iloc[:, i] = dummy
#     i += 1
#     dummy.clear()

# def sumar_sentimientos(df):
#
#     a= 0
#     b = 4
#     c = 8
#     lista = []
#     for i in range(4):
#         lista.extend([df.iloc[:3,[a,b,c]].values.sum(),df.iloc[3:7,[a,b,c]].values.sum(),df.iloc[7:,[a,b,c]].values.sum()])
#         a+=1
#         b+=1
#         c+=1
#     clasificacion_df = pd.DataFrame(index=['Textblob', 'Vader', 'SenticNet', 'LM'],
#                                     columns=['Clasificacion_perfecta', 'Clasificacion_equivocada',
#                                              'Clasificacion_contraria'], data=np.array(lista).reshape(4, 3))
#
#     return clasificacion_df

# df_sentimientos_metodos = sumar_sentimientos(df)


#
# for i, cm in enumerate(cms):
#     sns.heatmap(cm, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax.flat[i], annot=True)
#     ax.flat[i].set_title(cm_df.columns[i])
#     ax.flat[i].set_xlabel('Predicción')
#     ax.flat[i].set_ylabel('Real')
#     fig.suptitle('Matriz de confusion para noticias de Ecopetrol')
# plt.tight_layout()
# plt.show()
# # print(type(classification_report(eco_df.iloc[:, 0],cm_df.iloc[:, 0],labels=labels)))
# crs = [classification_report(eco_df.iloc[:, 0],cm_df.iloc[:, i], labels=labels, output_dict=True ) for i in range(len(cm_df.columns))]
# 'Araña larepublica '+str(empresas[0])+'.xlsx'

import pandas as pd
import datetime
import unidecode
import re
from textblob import TextBlob
from textblob.tokenizers import SentenceTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from senticnet.senticnet6 import senticnet
from nltk import word_tokenize

lm_dic_neg = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name='Negative', header=None)
lm_dic_pos = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name='Positive', header=None)

dic_titles = {}
dic_news = {}

empresas = ['ecopetrol', 'bancolombia', 'colcap']
news_title = True

noticias_writer_titulos = pd.ExcelWriter('noticias larepublica_title.xlsx', engine='xlsxwriter')

noticias_writer_completo = pd.ExcelWriter('noticias larepublica_completo.xlsx', engine='xlsxwriter')


if news_title == True:
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
    print('Ya estan los archivos traducidos?')
    confirmacion = input()
    if confirmacion == 'si':
        dic_titles = {}
        for i in range(len(empresas)):

            clasificadores = ['sentimiento_textblob', 'sentimiento_vader', 'sentimiento_senticnet', 'sentimiento_lm']

            dic_titles[empresas[i] + ' noticias'] = pd.read_excel('noticias larepublica_titulos.xlsx',
                                                                  sheet_name='{}_eng'.format(empresas[i]), index_col=0)
            dic_titles[empresas[i] + ' noticias'] = pd.Series(data=dic_titles[empresas[i] + ' noticias'].noticia,
                                                              index=dic_titles[empresas[i] + ' noticias'].index)

            for j in range(len(clasificadores)):
                dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[empresas[i] + ' noticias'].copy()
                if j == 0:
                    textblob_obj = [TextBlob(new) for new in dic_titles[empresas[i] + '_' + clasificadores[j]]]
                    polarity_textblob = [textblob_obj[i].sentiment.polarity for i in range(len(textblob_obj))]
                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = polarity_textblob


                    def sentimiento_t(polaridad):
                        if polaridad > 0.1:
                            sen = 1
                        elif polaridad < -0.05:
                            sen = -1
                        else:
                            sen = 0

                        return sen


                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_t(new) for new in
                                                                                 dic_titles[
                                                                                     empresas[i] + '_' + clasificadores[
                                                                                         j]]]
                    dic_titles[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                        data=dic_titles[empresas[i] + '_' + clasificadores[j]],
                        index=dic_titles[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
                    dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
                        empresas[i] + '_' + clasificadores[j]].groupby(level=0).sum()
                elif j == 1:
                    dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[empresas[i] + ' noticias'].copy()
                    vader_obj = SentimentIntensityAnalyzer()
                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [
                        vader_obj.polarity_scores(new)['compound'] for new in dic_titles[empresas[i] + ' noticias']]


                    def sentimiento_v(polaridad):
                        if polaridad > 0.05:
                            pol = 1
                        elif polaridad < -0.05:
                            pol = -1
                        else:
                            pol = 0

                        return pol


                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_v(pol) for pol in
                                                                                 dic_titles[
                                                                                     empresas[i] + '_' + clasificadores[
                                                                                         j]]]
                    dic_titles[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                        data=dic_titles[empresas[i] + '_' + clasificadores[j]],
                        index=dic_titles[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
                    dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
                        empresas[i] + '_' + clasificadores[j]].groupby(level=0).sum()
                elif j == 2:

                    sentiment_words = {}


                    def polarity_senticnet(list_of_words, dictionary):
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


                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [word_tokenize(text) for text in
                                                                                 dic_titles[
                                                                                     empresas[i] + '_' + clasificadores[
                                                                                         j]]]
                    stopwords = ['oil', 'production', 'crude', 'sector', 'energy', 'gas', 'pipeline', 'refinery',
                                 'petroleum', 'dollar',
                                 'infrastructure', 'water', 'hydrocarbon', 'latin', 'stocks', 'gasoline',
                                 'exploitation']
                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = dic_titles[
                                                                                    empresas[i] + '_' + clasificadores[
                                                                                        j]].iloc[:].apply(
                        lambda x: [word for word in x if word not in stopwords])
                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [
                        polarity_senticnet(text_list, senticnet)[0] for text_list in
                        dic_titles[empresas[i] + '_' + clasificadores[j]]]


                    def sentimiento_s(polaridad):
                        if polaridad > 0.35:
                            pol = 1
                        elif polaridad < 0.20:
                            pol = -1
                        else:
                            pol = 0

                        return pol


                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_s(pol) for pol in
                                                                                 dic_titles[
                                                                                     empresas[i] + '_' + clasificadores[
                                                                                         j]]]
                    dic_titles[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                        data=dic_titles[empresas[i] + '_' + clasificadores[j]],
                        index=dic_titles[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
                    dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
                        empresas[i] + '_' + clasificadores[j]].groupby(level=0).sum()
                else:
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
                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [word_tokenize(text) for text in
                                                                                 dic_titles[
                                                                                     empresas[i] + '_' + clasificadores[
                                                                                         j]]]
                    dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [lm_polarity(new, lm_dic)[0] for new in
                                                                                 dic_titles[
                                                                                     empresas[i] + '_' + clasificadores[
                                                                                         j]]]
                    dic_titles[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                        data=dic_titles[empresas[i] + '_' + clasificadores[j]],
                        index=dic_titles[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
                    dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
                        empresas[i] + '_' + clasificadores[j]].groupby(level=0).sum()

else:
    noticias_writer_completo = pd.ExcelWriter('noticias larepublica_completo.xlsx', engine='xlsxwriter')
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

    print('Ya estan los archivos traducidos?')
    confirmacion = input()
    if confirmacion == 'si':
        dic_news = {}
        for i in range(len(empresas)):

            clasificadores = ['sentimiento_textblob', 'sentimiento_vader', 'sentimiento_senticnet', 'sentimiento_lm']

            dic_news[empresas[i] + ' noticias'] = pd.read_excel('noticias larepublica.xlsx',
                                                                sheet_name='{}_eng'.format(empresas[i]), index_col=0)
            dic_news[empresas[i] + ' noticias'] = pd.Series(data=dic_news[empresas[i] + ' noticias'].noticia,
                                                            index=dic_news[empresas[i] + ' noticias'].index)

            for j in range(len(clasificadores)):
                dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[empresas[i] + ' noticias'].copy()
                if j == 0:
                    sentence_tokenizer = SentenceTokenizer()
                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentence_tokenizer.tokenize(text) for text in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[j]]]
                    textblob_obj = [[TextBlob(i) for i in paragr] for paragr in
                                    dic_news[empresas[i] + '_' + clasificadores[j]]]

                    sentences_polarities = [[textblob_obj[i][j].sentiment.polarity for j in range(len(textblob_obj[i]))] for i in
                                                                               range(len(textblob_obj))]



                    def blob_mod(list_p):
                        blob2_mod = []
                        for par in list_p:
                            if par < 0:
                                blob2_mod.append(par)
                            elif 0 <= par <= 0.25:
                                pass
                            else:
                                blob2_mod.append(par)
                        if len(blob2_mod) != 0:
                            blob2_mod = sum(blob2_mod) / len(blob2_mod)
                        else:
                            blob2_mod = 0

                        return blob2_mod


                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [blob_mod(text) for text in sentences_polarities]


                    def sentimiento_t(polaridad):
                        if polaridad > 0.1:
                            sen = 1
                        elif polaridad < -0.05:
                            sen = -1
                        else:
                            sen = 0

                        return sen


                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_t(new) for new in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[
                                                                                       j]]]
                    dic_news[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                        data=dic_news[empresas[i] + '_' + clasificadores[j]],
                        index=dic_news[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
                    dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[
                        empresas[i] + '_' + clasificadores[j]].groupby(level=0).sum()

                elif j == 1:

                    dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[empresas[i] + ' noticias'].copy()
                    vader_obj = SentimentIntensityAnalyzer()
                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentence_tokenizer.tokenize(text) for text in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[j]]]


                    def vader_modify(list_p):
                        vader_mod = []
                        for par in list_p:
                            if par < 0:
                                vader_mod.append(par)
                            elif 0 <= par <= 0.5:
                                pass
                            else:
                                vader_mod.append(par)
                        if len(vader_mod) != 0:
                            vader_mod = sum(vader_mod) / len(vader_mod)
                        else:
                            vader_mod = 0

                        return vader_mod

                    polaridad_vader = [
                        [vader_obj.polarity_scores(parragraph)['compound'] for parragraph in row] for row in
                        dic_news[empresas[i] + '_' + clasificadores[j]]]

                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [vader_modify(par) for par in polaridad_vader]



                    def sentimiento_v(polaridad):
                        if polaridad > 0.05:
                            pol = 1
                        elif polaridad < -0.05:
                            pol = -1
                        else:
                            pol = 0

                        return pol


                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_v(pol) for pol in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[
                                                                                       j]]]
                    dic_news[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                        data=dic_news[empresas[i] + '_' + clasificadores[j]],
                        index=dic_news[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
                    dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[
                        empresas[i] + '_' + clasificadores[j]].groupby(level=0).sum()

                elif j == 2:

                    sentiment_words = {}


                    def polarity_senticnet(list_of_words, dictionary):
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


                    stopwords = ['oil', 'production', 'crude', 'sector', 'energy', 'gas', 'pipeline', 'refinery',
                                 'petroleum', 'dollar',
                                 'infrastructure', 'water', 'hydrocarbon', 'latin', 'stocks', 'gasoline',
                                 'exploitation']
                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [word_tokenize(text) for text in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[
                                                                                       j]]]
                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = dic_news[
                                                                                  empresas[i] + '_' + clasificadores[
                                                                                      j]].iloc[:].apply(
                        lambda x: [word for word in x if word not in stopwords])

                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [
                        polarity_senticnet(text_list, senticnet)[0] for text_list in
                        dic_news[empresas[i] + '_' + clasificadores[j]]]


                    def sentimiento_s(polaridad):
                        if polaridad > 0.35:
                            pol = 1
                        elif polaridad < 0.30:
                            pol = -1
                        else:
                            pol = 0

                        return pol


                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_s(pol) for pol in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[
                                                                                       j]]]
                    dic_news[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                        data=dic_news[empresas[i] + '_' + clasificadores[j]],
                        index=dic_news[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
                    dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[
                        empresas[i] + '_' + clasificadores[j]].groupby(level=0).sum()

                else:
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
                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [word_tokenize(text) for text in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[
                                                                                       j]]]
                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [lm_polarity(new, lm_dic)[0] for new in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[
                                                                                       j]]]
                    dic_news[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                        data=dic_news[empresas[i] + '_' + clasificadores[j]],
                        index=dic_news[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
                    dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[
                        empresas[i] + '_' + clasificadores[j]].groupby(level=0).sum()

sheets_news = ['english_eco', 'english_banco', 'english_colcap']
sheets_financial = ['Ecopetrol OHLCV+indicadores', 'Bancolombia OHLCV+indicadores', 'Icolcap OHLCV+indicadores']
time = pd.date_range('2013-01-01', '2019-12-31', freq='D')
dic_dfs_f = pd.read_excel('OHLCV+indicadores.xlsx', index_col=0, sheet_name=sheets_financial)
dic_dfs_n = pd.read_excel('noticias larepublica.xlsx', index_col=0, sheet_name=sheets_news)

columns = list(dic_dfs_f['Ecopetrol OHLCV+indicadores'].columns)

df_eco_f = dic_dfs_f['Ecopetrol OHLCV+indicadores']
df_eco_f.insert(loc=8, column='retorno', value=0)
df_eco_f['retorno'] = df_eco_f.close - df_eco_f.close.shift(1)
df_eco_f['retorno'] = df_eco_f['retorno'].apply((lambda x: 1 if x > 0 else -1))

df_ban_f = dic_dfs_f['Bancolombia OHLCV+indicadores']
df_col_f = dic_dfs_f['Icolcap OHLCV+indicadores']

df_eco_n = dic_dfs_n['english_eco']
dates_e = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), df_eco_n.index))
df_eco_n.index = dates_e
df_eco_n.columns = ['noticias_ecopetrol']

df_ban_n = dic_dfs_n['english_banco']
dates_b = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), df_ban_n.index))
df_ban_n.index = dates_b
df_ban_n.columns = ['noticias_bancolombia']

df_col_n = dic_dfs_n['english_colcap']
dates_c = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), df_col_n.index))
df_col_n.index = dates_c
df_ban_n.columns = ['noticias_colcap']

df_time = pd.DataFrame(index=time)

# def add_by_index(df1,df2):
#     df = df1.copy()
#     idx = df1.index
#     for i,j in df2.iterrows():
#         try:
#             if i in idx:
#                 df.loc[i] = j
#             return df
#         except ValueError:
#             df.loc[i] = j
#             return df

senticnet_pol = pd.Series(senticnet_pol.noticia)
lm_pol = pd.Series(lm_pol.noticia)
df_polaridades = df.copy()
df_polaridades['polaridad_textblob'] = new_ss_textblob
df_polaridades['polaridad_vader'] = vader_eco_ss_d
df_polaridades['polaridad_senticnet'] = senticnet_pol
df_polaridades['polaridad_lm'] = lm_pol


def sentimiento_t(polaridad):
    if polaridad > 0.1:
        pol = 1
    elif polaridad < -0.05:
        pol = -1
    else:
        pol = 0

    return pol


def sentimiento_v(polaridad):
    if polaridad > 0.05:
        pol = 1
    elif polaridad < -0.05:
        pol = -1
    else:
        pol = 0

    return pol


def sentimiento_s(polaridad):
    if polaridad > 0.35:
        pol = 1
    elif polaridad < 0.30:
        pol = -1
    else:
        pol = 0

    return pol


def sentimiento_lm(polaridad):
    if polaridad > 0:
        pol = 1
    elif polaridad < 0:
        pol = -1
    else:
        pol = 0

    return pol


df_polaridades['polaridad_textblob'] = [sentimiento_t(pol) for pol in df_polaridades['polaridad_textblob']]
df_polaridades['polaridad_vader'] = [sentimiento_v(pol) for pol in df_polaridades['polaridad_vader']]
df_polaridades['polaridad_senticnet'] = [sentimiento_s(pol) for pol in df_polaridades['polaridad_senticnet']]
df_polaridades['polaridad_lm'] = [sentimiento_lm(pol) for pol in df_polaridades['polaridad_lm']]

s_polaridad_textblob = pd.Series(df_polaridades['polaridad_textblob'], dtype='float')
s_polaridad_textblob = s_polaridad_textblob.groupby(level=0).sum()

s_polaridad_vader = pd.Series(df_polaridades['polaridad_vader'], dtype='float')
s_polaridad_vader = s_polaridad_vader.groupby(level=0).sum()

s_polaridad_senticnet = pd.Series(df_polaridades['polaridad_senticnet'], dtype='float')
s_polaridad_senticnet = s_polaridad_senticnet.groupby(level=0).sum()

s_polaridad_lm = pd.Series(df_polaridades['polaridad_lm'], dtype='float')
s_polaridad_lm = s_polaridad_lm.groupby(level=0).sum()
dic_df = {'polaridad_textblob': s_polaridad_textblob, 'polaridad_vader': s_polaridad_vader,
          'polaridad_senticnet': s_polaridad_senticnet, 'polaridad_lm': s_polaridad_lm}
df_polaridades = pd.DataFrame(dic_df)
# idx_list = s_polaridad_textblob.index
#
# int_idx_list = [int(idx_list[i].strftime('%Y%m%d')) for i in range(len(idx_list))]
# s_polaridad_textblob_idx_str = s_polaridad_textblob.copy()
# s_polaridad_textblob_idx_str.index = int_idx_list
# s_polaridad_textblob_copy = s_polaridad_textblob_idx_str.copy()
# i=0
# index_previous = 0

# for index, row in s_polaridad_textblob.iteritems():
#
#
#     if i == 0:
#         s_polaridad_textblob_copy.iloc[i] = row * 0.75
#         i +=1
#         index_previous = index
#     elif index - index_previous == 1:
#         s_polaridad_textblob_copy.iloc[i] = s_polaridad_textblob_copy.iloc[i-1]*0.25 + row*0.75
#         i += 1
#         index_previous = index
#
#     else:
#         s_polaridad_textblob_copy.iloc[i] = row * 0.75
#         i += 1
#         index_previous = index


# formato_fecha = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), df_polaridades.index))
# df_polaridades.index = formato_fecha
# df_eco_c = add_by_index(df_time,df_eco)
df_eco_m = df_time.merge(df_eco_f, left_index=True, right_index=True, how='left')
df_eco_fn_m = df_eco_m.merge(df_polaridades, left_index=True, right_index=True, how='left')
df_eco_fn_m_c = df_eco_fn_m.copy()


def sentimientos_finales(*series):
    clasificadores = ['sentimiento_textblob', 'sentimiento_vader', 'sentimiento_senticnet', 'sentimiento_lm']
    df_sentimientos = pd.DataFrame()

    for j, serie in enumerate(series):
        serie = serie.fillna(0)
        serie_test = serie.copy()
        i = 0

        for index, row in serie.iteritems():
            try:

                if i == 0:
                    serie_test.iloc[i] = row
                    i += 1
                else:
                    serie_test.iloc[i] = serie.iloc[i - 1] * 0.25 + row * 0.75
                    i += 1
                    if -0.05 < serie_test.iloc[i - 1] < 0.05:
                        serie_test.iloc[i] = 0

            except IndexError:
                pass
        df_sentimientos[clasificadores[j]] = serie_test

    def escalar(x):
        if x > 0:
            x = 1
        elif x < 0:
            x = -1
        else:
            x = 0
        return x

    for i in range(len(df_sentimientos.columns)):
        df_sentimientos[clasificadores[i]] = list(map(escalar, df_sentimientos.iloc[:, i]))

    return df_sentimientos


v = sentimientos_finales(df_eco_fn_m.iloc[:, 9], df_eco_fn_m.iloc[:, 10], df_eco_fn_m.iloc[:, 11],
                         df_eco_fn_m.iloc[:, 12])
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report

df_concat = pd.concat([df_eco_fn_m_c,
                       sentimientos_finales(df_eco_fn_m.iloc[:, 9], df_eco_fn_m.iloc[:, 10], df_eco_fn_m.iloc[:, 11],
                                            df_eco_fn_m.iloc[:, 12])], axis=1)
df_concat = df_concat.drop(axis=1, columns=df_polaridades.columns)
df_concat = df_concat.loc['2016-03-01':'2019-12-31']


def sentiment_final_metrics(df):
    clasificadores = ['sentimiento_textblob', 'sentimiento_vader', 'sentimiento_senticnet', 'sentimiento_lm']
    started_column = 9
    df_sfm = pd.DataFrame()
    dic_metrics = {}
    for i in range(len(clasificadores)):
        df_sfm = df.iloc[:, [8, started_column]]
        df_sfm = df_sfm.dropna()
        df_sfm = df_sfm[df_sfm.iloc[:, 1] != 0]
        dic_metrics['accuracy_' + str(clasificadores[i])] = accuracy_score(df_sfm.iloc[:, 0], df_sfm.iloc[:, 1])
        dic_metrics['confusion_matrix_' + str(clasificadores[i])] = confusion_matrix(df_sfm.iloc[:, 0],
                                                                                     df_sfm.iloc[:, 1])
        dic_metrics['classification_report_' + str(clasificadores[i])] = classification_report(df_sfm.iloc[:, 0],
                                                                                               df_sfm.iloc[:, 1])
        started_column += 1

    return dic_metrics

# df_r_t = df.iloc[:,[8,9]]
# df_r_t= df_r_t.dropna()
# df_r_t = df_r_t[df_r_t.iloc[:,1] != 0]
# cm = confusion_matrix(df_r_t.iloc[:,0], df_r_t.iloc[:,1], labels=[-1,1])
# acc = accuracy_score(df_r_t.iloc[:,0],df_r_t.iloc[:,1])
# cr = classification_report(df_r_t.iloc[:,0], df_r_t.iloc[:,1])

#
# f_r_t = df_concat.iloc[:,[8,9]]
# df_r_t= df_r_t.dropna()
# df_r_t = df_r_t[df_r_t.iloc[:,1] != 0]
# cm = confusion_matrix(df_r_t.iloc[:,0], df_r_t.iloc[:,1], labels=[-1,1])
# acc = accuracy_score(df_r_t.iloc[:,0],df_r_t.iloc[:,1])
# cr = classification_report(df_r_t.iloc[:,0], df_r_t.iloc[:,1])


# df_eco_fn_m['polaridad_textblob'] = df_eco_fn_m['polaridad_textblob'].fillna(0)
# df_eco_fn_m['polaridad_textblob_test'] = 0
# i=0
# index_previous = 0
# for index, row in df_eco_fn_m['polaridad_textblob'].iteritems():
#     if i == 0:
#         df_eco_fn_m['polaridad_textblob_test'].iloc[i] = row
#         i+=1
#     else:
#         df_eco_fn_m['polaridad_textblob_test'].iloc[i] = df_eco_fn_m['polaridad_textblob'].iloc[i-1]*0.25 + row * 0.75
#         i += 1
#         if -0.05 < df_eco_fn_m['polaridad_textblob_test'].iloc[i-1] < 0.05:
#             df_eco_fn_m['polaridad_textblob_test'].iloc[i] = 0
