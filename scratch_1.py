import pandas as pd
import statsmodels.discrete.discrete_model as sm
import seaborn as sns
import matplotlib.pyplot as plt
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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle
import datetime as dt
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


dic = {'a':['hola']}

dic['a']['chao'] = 2

