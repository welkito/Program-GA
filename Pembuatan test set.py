import pandas as pd
import numpy as np
import string
import csv
import re
import pickle
from random import randint
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from nltk import tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
# import ga

datadir = 'C:/Users/Wellington/Desktop/skripC/processing/'
############################################################################################################################################################
test = pd.read_pickle(datadir+"dfdatalabelcleanser.pkl")
test = test.reset_index()
test = test.drop('index', axis = 1)
# test = test.drop('level_0', axis = 1)
data = test['FIXBERSIH'].values.tolist()
dlabel = []

for kiri,kanan in test['Label'].items() :
    dlabel.append(kanan)

indeks = 0
nilai = 0
while indeks < len(data) :
    if (data[indeks] == []) :
        data.remove(data[indeks])
        dlabel.remove(dlabel[indeks])
        nilai+=1
        print(nilai)
    else :
        indeks += 1

Ratio = pd.read_pickle('C:/Users/Wellington/Desktop/skripC/processing/FeatureRasioSortircleanser.pkl')
Ratio = Ratio.reset_index()
Rasio = Ratio.drop('index', axis = 1)

vocab = Rasio["Feature"].values.tolist()

row =[]
isi=[]

for b in range(len(vocab)):
    isi.append(0)

print(data.__len__())
for a in range(len(data)):
   row.append(isi)

datarow= np.array(row)#buat dari list jadi array, jadi pecah isinya jadi array
objek = np.array(data)#melihat data's list untukdipecah jadi array, dimana nilai array sekarang per kata
tes = pd.DataFrame(datarow,columns = [vocab])

for p_data in range(data.__len__()): #panjang jumlah review (data)>> hitung kebawah
    print("iterasii -",p_data)
    for p_kolom in range(vocab.__len__()): #panjang colom, dibawahnya masukkin nilai array review berikutnya (vocab)>> ke kanan
        for p_objek in range(data[p_data].__len__()) : #panjang masing masing array(len(data[i]))
            if objek[p_data][p_objek] == vocab[p_kolom] :
                tes.loc[p_data][p_kolom] = tes.loc[p_data][p_kolom] + 1

tes['Label'] = dlabel.copy()
tes.to_csv(r'C:/Users/Wellington/Desktop/skripC/processing/dataframecleanser.csv',index = False, header=True)
tes.to_pickle(r'C:/Users/Wellington/Desktop/skripC/processing/dataframecleanser.pkl')
