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
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
lm_dic_neg = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name='Negative', header=None)
lm_dic_pos = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name='Positive', header=None)

# ## DEBIDO A QUE SON 3 DATASETS( 3 ACCIONES A EVALUAR) Y 2 INSTANCIAS(TITULO, Y TITULO+TEXTO), SE ORGANIZA LA
# INFORMACION EN DOS DICCIONARIOS
dic_titles = {}
dic_news = {}

empresas = ['ecopetrol', 'bancolombia', 'colcap']


noticias_writer_titulos = pd.ExcelWriter('noticias larepublica_title.xlsx', engine='xlsxwriter')
noticias_writer_completo = pd.ExcelWriter('noticias larepublica_completo.xlsx', engine='xlsxwriter')


### INSTANCIA DE NOTICIAS UNICAMENTE CON TITULO

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
            polarity_textblob = [textblob_obj[i].sentiment.polarity if textblob_obj[i].sentiment.polarity ==0 else textblob_obj[i].sentiment.polarity-0.1 for i in range(len(textblob_obj))]
            dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = polarity_textblob


            dic_titles[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                data=dic_titles[empresas[i] + '_' + clasificadores[j]],
                index=dic_titles[empresas[i] + '_' + clasificadores[j]].index, dtype='float')

            dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
                empresas[i] + '_' + clasificadores[j]].groupby(level=0).mean()

        elif j == 1:
            dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[empresas[i] + ' noticias'].copy()
            vader_obj = SentimentIntensityAnalyzer()
            dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [
                vader_obj.polarity_scores(new)['compound'] if vader_obj.polarity_scores(new)['compound'] == 0 else vader_obj.polarity_scores(new)['compound']-0.05  for new in dic_titles[empresas[i] + ' noticias']]


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
                polarity_senticnet(text_list, senticnet)[0] if polarity_senticnet(text_list, senticnet)[0] == 0 else polarity_senticnet(text_list, senticnet)[0]-0.35 for text_list in
                dic_titles[empresas[i] + '_' + clasificadores[j]]]



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
            dic_titles[empresas[i] + '_' + clasificadores[j]].iloc[:] = [lm_polarity(new, lm_dic)[0] for new in
                                                                         dic_titles[
                                                                             empresas[i] + '_' + clasificadores[
                                                                                 j]]]


            dic_titles[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                data=dic_titles[empresas[i] + '_' + clasificadores[j]],
                index=dic_titles[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
            dic_titles[empresas[i] + '_' + clasificadores[j]] = dic_titles[
                empresas[i] + '_' + clasificadores[j]].groupby(level=0).mean()


## INSTANCIA DE NOTICIAS COMPLETAS (CUERPO + TITULO)


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


            dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [blob_mod(text)-0.05 for text in
                                                                       sentences_polarities]

            dic_news[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                data=dic_news[empresas[i] + '_' + clasificadores[j]],
                index=dic_news[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
            dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[
                empresas[i] + '_' + clasificadores[j]].groupby(level=0).mean()

        elif j == 1:

            dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[empresas[i] + ' noticias'].copy()
            vader_obj = SentimentIntensityAnalyzer()
            sentence_tokenizer = SentenceTokenizer()
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

            dic_news[empresas[i] + '_' + clasificadores[j]].iloc[:] = [vader_modify(par)-0.05 for par in
                                                                       polaridad_vader]


            dic_news[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                data=dic_news[empresas[i] + '_' + clasificadores[j]],
                index=dic_news[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
            dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[
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
                polarity_senticnet(text_list, senticnet)[0]-0.35 for text_list in
                dic_news[empresas[i] + '_' + clasificadores[j]]]


            dic_news[empresas[i] + '_' + clasificadores[j]] = pd.Series(
                data=dic_news[empresas[i] + '_' + clasificadores[j]],
                index=dic_news[empresas[i] + '_' + clasificadores[j]].index, dtype='float')
            dic_news[empresas[i] + '_' + clasificadores[j]] = dic_news[
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
                empresas[i] + '_' + clasificadores[j]].groupby(level=0).mean()

dic_df_sentimientos = {'df_ecopetrol_completo': pd.DataFrame({'sentimiento_textblob':dic_news['ecopetrol_sentimiento_textblob'].iloc[:], 'sentimiento_vader': dic_news['ecopetrol_sentimiento_vader'].values, 'sentimiento_senticnet': dic_news['ecopetrol_sentimiento_senticnet'].values, 'sentimiento_lm':dic_news['ecopetrol_sentimiento_lm'].values}),
                       'df_bancolombia_completo': pd.DataFrame({'sentimiento_textblob':dic_news['bancolombia_sentimiento_textblob'], 'sentimiento_vader': dic_news['bancolombia_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_news['bancolombia_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_news['bancolombia_sentimiento_lm'].iloc[:]}),
                       'df_icolcap_completo': pd.DataFrame({'sentimiento_textblob':dic_news['colcap_sentimiento_textblob'], 'sentimiento_vader': dic_news['colcap_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_news['colcap_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_news['colcap_sentimiento_lm'].iloc[:]}),
                       'df_ecopetrol_titulo': pd.DataFrame({'sentimiento_textblob':dic_titles['ecopetrol_sentimiento_textblob'], 'sentimiento_vader': dic_titles['ecopetrol_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_titles['ecopetrol_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_titles['ecopetrol_sentimiento_lm'].iloc[:]}),
                       'df_bancolombia_titulo': pd.DataFrame({'sentimiento_textblob':dic_titles['bancolombia_sentimiento_textblob'], 'sentimiento_vader': dic_titles['bancolombia_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_titles['bancolombia_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_titles['bancolombia_sentimiento_lm'].iloc[:]}),
                       'df_icolcap_titulo': pd.DataFrame({'sentimiento_textblob':dic_titles['colcap_sentimiento_textblob'], 'sentimiento_vader': dic_titles['colcap_sentimiento_vader'].iloc[:], 'sentimiento_senticnet': dic_titles['colcap_sentimiento_senticnet'].iloc[:], 'sentimiento_lm':dic_titles['colcap_sentimiento_lm'].iloc[:]})}

with open('pickle_sentimientos_mod.pkl', 'wb') as file:
    pickle.dump(dic_df_sentimientos, file)