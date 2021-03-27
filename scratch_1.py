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
        j = 3
        dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[empresas[i] + ' noticias'].copy()
        if j == 0:
            textblob_obj = [TextBlob(new) for new in dic_titles[empresas[i] + '_' + clasificadores[j]]]
            polarity_textblob = [textblob_obj[i].sentiment.polarity for i in range(len(textblob_obj))]
            dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = polarity_textblob


            # def sentimiento_t(polaridad):
            #     if polaridad > 0.1:
            #         sen = 1
            #     elif polaridad < -0.05:
            #         sen = -1
            #     else:
            #         sen = 0
            #
            #     return sen


            # dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_t(new) for new in
            #                                                              dic_titles[
            #                                                                  empresas[i] + '_' + clasificadores[
            #                                                                      j]]]
            dic_titles[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                data=dic_titles[empresas[i] + '_' + clasificadores[j]],
                index=dic_titles[empresas[i] + '_' + clasificadores[j]].index, dtype='float')

            dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
                empresas[i] + '_' + clasificadores[j]].groupby(level=0).mean()

        elif j == 1:
            dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[empresas[i] + ' noticias'].copy()
            vader_obj = SentimentIntensityAnalyzer()
            dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [
                vader_obj.polarity_scores(new)['compound'] for new in dic_titles[empresas[i] + ' noticias']]


            # def sentimiento_v(polaridad):
            #     if polaridad > 0.05:
            #         pol = 1
            #     elif polaridad < -0.05:
            #         pol = -1
            #     else:
            #         pol = 0
            #
            #     return pol


            # dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_v(pol) for pol in
            #                                                              dic_titles[
            #                                                                  empresas[i] + '_' + clasificadores[
            #                                                                      j]]]
            dic_titles[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                data=dic_titles[empresas[i] + '_' + clasificadores[j]],
                index=dic_titles[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
            dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
                empresas[i] + '_' + clasificadores[j]].groupby(level=0).mean()
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
                empresas[i] + '_' + clasificadores[j]].groupby(level=0).mean()
        elif j == 3:

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
            dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [lm_polarity(new, lm_dic) for new in
                                                                         dic_titles[
                                                                             empresas[i] + '_' + clasificadores[
                                                                                 j]]]


            # def sentimiento_lm(polaridad):
            #     if polaridad > 0:
            #         pol = 1
            #     elif polaridad < 0:
            #         pol = -1
            #     else:
            #         pol = 0
            #
            #     return pol
            #
            #
            # dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [sentimiento_lm(new) for new in
            #                                                            dic_titles[
            #                                                                empresas[i] + '_' + clasificadores[
            #                                                                    j]]]


            # dic_titles[empresas[i] + '_' + clasificadores[j]] = pd.Series(
            #     data=dic_titles[empresas[i] + '_' + clasificadores[j]],
            #     index=dic_titles[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
            # dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
            #     empresas[i] + '_' + clasificadores[j]].groupby(level=0).mean()

        # elif j == 4:
        #     dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = 0
        #     dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
        #         empresas[i] + '_' + clasificadores[j]].groupby(level=0).sum()
        #
        #     dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [float(np.random.randint(-1,2)) for i in dic_titles[empresas[i] + '_' + clasificadores[j]]]



