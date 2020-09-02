import pandas as pd
import numpy as np
# import string
import csv
# import re
import pickle
import math
# from imblearn.metrics import geometric_mean_score
# from random import randint
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# import nltk
# from nltk.tokenize.treebank import TreebankWordDetokenizer
# from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
# from sklearn import svm
# from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt
import matplotlib
# import warnings
# warnings.filterwarnings('error')

datadir = 'C:/Users/Wellington/Desktop/skripC/processing/'

Ratio = pd.read_pickle('C:/Users/Wellington/Desktop/skripC/processing/FeatureRasioSortircleanser.pkl')
Ratio = Ratio.reset_index()
Rasio = Ratio.drop('index', axis = 1)
#
# vocab = Rasio["Feature"].values.tolist()
# vocab.append("Label")
# df= pd.read_pickle(r'C:/Users/Wellington/Desktop/skripC/processing/dataframecleanser.pkl')
# awal = pd.DataFrame(df, columns = [vocab] )
# awal= awal.sample(frac=1).reset_index(drop=True)
# x = pd.DataFrame(awal.drop('Label', axis = 1))
# y = awal[['Label']].values.copy()
# vocab.remove(vocab[len(vocab)-1])
# x_yes = pd.DataFrame(columns = [vocab])
# x_no = pd.DataFrame(columns = [vocab])
# atas = 0
# indeks = -1
# print(len(y))
# ############################################################################
# for a in y :
#     indeks += 1
#     print(indeks)
#     if a == 1 :
#         x_yes = x_yes.append(x.loc[indeks], ignore_index = True)
#     if a == 0 :
#         x_no = x_no.append(x.loc[indeks], ignore_index = True)
#         atas += 1
#         print(atas)
#     if atas == 3125:
#         break
# x_tengah = pd.DataFrame(columns = [vocab])
# y_tengah = []
# for i in range(3125):
#     x_tengah = x_tengah.append(x_yes.loc[i], ignore_index = True)
#     y_tengah.append(1)
# for i in range(3125):
#     x_tengah = x_tengah.append(x_no.loc[i], ignore_index = True)
#     y_tengah.append(0)
# print("panjang dari x : ",len(x))
# print("panjang dari x_tengah",len(x_tengah))
# x = x_tengah.copy()
# y = y_tengah.copy()
# x_tengah.to_pickle(r'C:/Users/Wellington/Desktop/skripC/processing/x_fix_cleanser.pkl')
# pickle.dump(y_tengah,open(r'C:/Users/Wellington/Desktop/skripC/processing/y_fix_cleanser.pkl',"wb"))
# ######################################################################################
#
# # ###################################################################################################
accuracyscores = []
crossmean =[]
d = range(1,60)
######################################################################################################
# tengah = len(x)/2
# print("total yes awal :  ",atas)
#
# over = RandomOverSampler(sampling_strategy = tengah/atas )
# under = RandomUnderSampler(sampling_strategy="majority")
# x_over, y_over = over.fit_resample(x, y)
# x_under,y_under = under.fit_resample(x_over,y_over)
#
# x_under.to_pickle(r'C:/Users/Wellington/Desktop/skripC/processing/xunder.pkl')
# pickle.dump(y_under,open(r'C:/Users/Wellington/Desktop/skripC/processing/yunder.pkl',"wb"))
x = pd.read_pickle(r'C:/Users/Wellington/Desktop/skripC/processing/x_fix_cleanser.pkl')
print(len(x))
y= pickle.load(open(r'C:/Users/Wellington/Desktop/skripC/processing/y_fix_cleanser.pkl',"rb"))
print(len(y))
###########################################################################################################
for bey in d:
    print("iterasi",bey)
    model = DecisionTreeClassifier(max_depth = bey)
    cross_score = cross_val_score(model, x.to_numpy(), y, cv=10)
    print("udah")
    crossmean.append(cross_score.mean())
fig, ax = plt.subplots()
ax.plot(d, crossmean, '-o', label='mean cross-validation accuracy', alpha=0.9)
ax.set_title("hasil maxdepth CV decision Tree /'cleanser/' review : ", fontsize=16)
ax.set_xlabel('Tree depth', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
