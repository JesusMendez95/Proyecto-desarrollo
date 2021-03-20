import pandas as pd
import datetime
import random as rd
import unidecode
import re
from textblob import TextBlob
from textblob.tokenizers import SentenceTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from senticnet.senticnet6 import senticnet
from nltk import word_tokenize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
lm_dic_neg = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name='Negative', header=None)
lm_dic_pos = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name='Positive', header=None)

dic_titles = {}
dic_news = {}

empresas = ['ecopetrol', 'bancolombia', 'colcap']


noticias_writer_titulos = pd.ExcelWriter('noticias larepublica_title.xlsx', engine='xlsxwriter')
noticias_writer_completo = pd.ExcelWriter('noticias larepublica_completo.xlsx', engine='xlsxwriter')
instancia_estudio = [True,False] # Si es True el estudio tiene en cuenta los titulares de las noticias, False el texto completo
for instancia in instancia_estudio:

    if instancia == True:
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
                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentence_tokenizer.tokenize(text) for
                                                                               text in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[
                                                                                       j]]]
                    textblob_obj = [[TextBlob(i) for i in paragr] for paragr in
                                    dic_news[empresas[i] + '_' + clasificadores[j]]]

                    sentences_polarities = [[textblob_obj[i][j].sentiment.polarity for j in range(len(textblob_obj[i]))]
                                            for i in
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


                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [blob_mod(text) for text in
                                                                               sentences_polarities]


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
                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentence_tokenizer.tokenize(text) for
                                                                               text in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[
                                                                                       j]]]


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

                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [vader_modify(par) for par in
                                                                               polaridad_vader]


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


                    def sentimiento_lm(polaridad):
                        if polaridad > 0:
                            pol = 1
                        elif polaridad < 0:
                            pol = -1
                        else:
                            pol = 0

                        return pol


                    dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_lm(new) for new in
                                                                               dic_news[
                                                                                   empresas[i] + '_' + clasificadores[
                                                                                       j]]]






dic_df_sentimientos = {'df_ecopetrol_completo': pd.DataFrame({'sentimiento_textblob':dic_news['ecopetrol_sentimiento_textblob'].iloc[:], 'sentimiento_vader': dic_news['ecopetrol_sentimiento_vader'].values, 'sentimiento_senticnet': dic_news['ecopetrol_sentimiento_senticnet'].values, 'sentimiento_lm':dic_news['ecopetrol_sentimiento_lm'].values}),
                       'df_bancolombia_completo': pd.DataFrame({'sentimiento_textblob':dic_news['bancolombia_sentimiento_textblob'], 'sentimiento_vader': dic_news['bancolombia_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_news['bancolombia_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_news['bancolombia_sentimiento_lm'].iloc[:]}),
                       'df_colcap_completo': pd.DataFrame({'sentimiento_textblob':dic_news['colcap_sentimiento_textblob'], 'sentimiento_vader': dic_news['colcap_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_news['colcap_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_news['colcap_sentimiento_lm'].iloc[:]}),
                       'df_ecopetrol_titulo': pd.DataFrame({'sentimiento_textblob':dic_titles['ecopetrol_sentimiento_textblob'], 'sentimiento_vader': dic_titles['ecopetrol_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_titles['ecopetrol_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_titles['ecopetrol_sentimiento_lm'].iloc[:]}),
                       'df_bancolombia_titulo': pd.DataFrame({'sentimiento_textblob':dic_titles['bancolombia_sentimiento_textblob'], 'sentimiento_vader': dic_titles['bancolombia_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_titles['bancolombia_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_titles['bancolombia_sentimiento_lm'].iloc[:]}),
                       'df_colcap_titulo': pd.DataFrame({'sentimiento_textblob':dic_titles['colcap_sentimiento_textblob'], 'sentimiento_vader': dic_titles['colcap_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_titles['colcap_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_titles['colcap_sentimiento_lm'].iloc[:]})}

empresas = ['ecopetrol', 'bancolombia', 'colcap']
clasificadores = ['sentimiento_textblob', 'sentimiento_vader', 'sentimiento_senticnet', 'sentimiento_lm']
sheets_financial = ['Ecopetrol OHLCV+indicadores', 'Bancolombia OHLCV+indicadores', 'Icolcap OHLCV+indicadores']

time = pd.date_range('2013-01-01', '2019-12-31', freq='D')
df_time = pd.DataFrame(index=time)
dic_financiero = {}
dic_financiero_copy = {}
dic_financiero_titulo = {}
dic_financiero_completo = {}
instancia_estudio = ['_titulo','_completo']

for instancia in instancia_estudio:

    tipo_de_instancia = instancia


    for i in range(len(sheets_financial)):
        dic_financiero[empresas[i]+ '_df'] = pd.read_excel('OHLCV+indicadores.xlsx', sheet_name=sheets_financial[i], index_col=0)

        dic_financiero[empresas[i] + '_df'].insert(loc=8, column='retorno', value=0)
        dic_financiero[empresas[i] + '_df']['retorno'] = dic_financiero[empresas[i] + '_df']['close'] - dic_financiero[empresas[i] + '_df']['close'].shift(1)
        dic_financiero[empresas[i] + '_df']['retorno'] = dic_financiero[empresas[i] + '_df']['retorno'].apply(lambda  x: 1 if x >0 else -1)
        dic_financiero[empresas[i] + '_df'] = df_time.merge(dic_financiero[empresas[i] + '_df'],left_index=True, right_index=True, how='left')
        dic_financiero_copy[empresas[i] + '_df'] =dic_financiero[empresas[i] + '_df'].copy()
        dic_financiero_copy[empresas[i] + '_df'] = dic_financiero_copy[empresas[i] + '_df'].merge(dic_df_sentimientos['df_'+empresas[i]+tipo_de_instancia], left_index=True, right_index=True, how='left')


        def sentimientos_finales(*series):
            clasificadores = ['sentimiento_textblob', 'sentimiento_vader', 'sentimiento_senticnet',
                              'sentimiento_lm']
            df_sentimientos = pd.DataFrame(dtype='float')

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



        dic_financiero_copy[empresas[i] + '_df'] = pd.concat([dic_financiero_copy[empresas[i] + '_df'], sentimientos_finales(dic_financiero_copy[empresas[i] + '_df'][clasificadores[0]],dic_financiero_copy[empresas[i] + '_df'][clasificadores[1]],dic_financiero_copy[empresas[i] + '_df'][clasificadores[2]],dic_financiero_copy[empresas[i] + '_df'][clasificadores[3]])], axis=1)
        dic_financiero[empresas[i] + '_df'] = pd.concat([dic_financiero[empresas[i] + '_df'], dic_financiero_copy[empresas[i] + '_df'].iloc[:,[13,14,15,16]]], axis=1)
        dic_financiero_copy[empresas[i] + '_df'] = dic_financiero[empresas[i] + '_df'].copy()
        dic_financiero_copy[empresas[i] + '_df'] = dic_financiero_copy[empresas[i] + '_df'][np.isnan(dic_financiero_copy[empresas[i] + '_df'].retorno) == False]
        dic_financiero_copy[empresas[i] + '_df'] = dic_financiero_copy[empresas[i] + '_df'][(dic_financiero_copy[empresas[i] + '_df'].iloc[:,9] != 0) & (dic_financiero_copy[empresas[i] + '_df'].iloc[:,10] !=0) & (dic_financiero_copy[empresas[i] + '_df'].iloc[:,11] != 0) & (dic_financiero_copy[empresas[i] + '_df'].iloc[:,12] != 0)]
        dic_financiero[empresas[i] + '_df'] = dic_financiero_copy[empresas[i] + '_df'].copy()
        dic_financiero[empresas[i] + '_df'] = dic_financiero[empresas[i] + '_df'].loc['2016-03-01':'2019-12-31']

        if instancia == '_titulo':
            dic_financiero_titulo[empresas[i] + '_df'] = dic_financiero[empresas[i] + '_df'].copy()
        elif instancia == '_completo':
            dic_financiero_completo[empresas[i] + '_df'] = dic_financiero[empresas[i] + '_df'].copy()


# Metricas generales
dic_metrics_completo = {}
dic_metrics_titulo = {}
dic_metrics = {}
for instancia in instancia_estudio:
    if instancia == '_titulo':
        dic_financiero = dic_financiero_titulo
    elif instancia == '_completo':
        dic_financiero = dic_financiero_completo
    for i in range(len(empresas)):

        for j in range(len(clasificadores)):
            dic_metrics['confusion_matrix_' + empresas[i]+'_'+clasificadores[j]] = confusion_matrix(dic_financiero[empresas[i]+'_df'].retorno,dic_financiero[empresas[i]+'_df'][clasificadores[j]])
            dic_metrics['classification_report_'+empresas[i]+'_'+clasificadores[j]] = classification_report(dic_financiero[empresas[i]+'_df'].retorno,dic_financiero[empresas[i]+'_df'][clasificadores[j]], output_dict=True)
            if instancia == '_titulo':
                dic_metrics_titulo['confusion_matrix_' + empresas[i]+'_'+clasificadores[j]] = dic_metrics['confusion_matrix_' + empresas[i]+'_'+clasificadores[j]]
                dic_metrics_titulo['classification_report_' + empresas[i] + '_' + clasificadores[j]] = dic_metrics['classification_report_'+empresas[i]+'_'+clasificadores[j]]
            elif instancia == '_completo':
                dic_metrics_completo['confusion_matrix_' + empresas[i] + '_' + clasificadores[j]] = dic_metrics['confusion_matrix_' + empresas[i] + '_' + clasificadores[j]]
                dic_metrics_completo['classification_report_' + empresas[i] + '_' + clasificadores[j]] = dic_metrics['classification_report_'+empresas[i]+'_'+clasificadores[j]]

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns',
              1000)  # Setup para ampliar visualización de datos (en este caso DataFrame) en el IDE
pd.set_option('display.width', 1000)
df_metricas = pd.read_excel('df_metricas.xlsx', index_col=[0,1], header =[0,1])



print(df_metricas.iloc[[0,1],:])


df_metricas = df_metricas.T
print(df_metricas)
df_metricas.iloc[0,0] = 555
k = 0
estudio_instancias= ['_completo', '_titulo']

for instancia in estudio_instancias:

    if instancia == '_titulo':
        dic_metrics = dic_metrics_titulo
    elif instancia == '_completo':
        dic_metrics = dic_metrics_completo
    for i in range(len(empresas)):

        for j in range(len(clasificadores)):

            df_metricas.iloc[0,k] = dic_metrics['classification_report_'+empresas[i]+'_'+clasificadores[j]]['accuracy']
            df_metricas.iloc[1, k] = dic_metrics['classification_report_' + empresas[i] + '_' + clasificadores[j]][
                '1.0']['precision']
            df_metricas.iloc[2, k] = dic_metrics['classification_report_' + empresas[i] + '_' + clasificadores[j]][
                '-1.0']['precision']
            df_metricas.iloc[3, k] = dic_metrics['classification_report_' + empresas[i] + '_' + clasificadores[j]][
                '1.0']['recall']
            df_metricas.iloc[4, k] = dic_metrics['classification_report_' + empresas[i] + '_' + clasificadores[j]][
                '-1.0']['recall']
            df_metricas.iloc[5, k] = dic_metrics['classification_report_' + empresas[i] + '_' + clasificadores[j]][
                'weighted avg']['f1-score']

            k+=1













