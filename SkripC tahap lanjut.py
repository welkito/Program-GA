import pandas as pd
import numpy as np
import string
import csv
import re
import pickle
import math
import random
from tabulate import tabulate
from datetime import datetime as dt
# from imblearn.metrics import geometric_mean_score
from sklearn.tree.export import export_text
from random import randint
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from nltk import tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import matplotlib
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
# import ga
import graphviz
datadir = 'C:/Users/Wellington/Desktop/skripC/processing/'
############################################################################################################################################################
train = pd.read_pickle(datadir+"train.pkl")

train = train.reset_index()
train = train.drop('index', axis = 1)
data = train['FIXBERSIH'].values.tolist()
# data = pickle.load(open(r'C:/Users/Wellington/Desktop/skripC/processing/data.pkl',"rb"))
# dlabel = []
#
# for kiri,kanan in train['Label'].items() :
#     dlabel.append(kanan)
#
# indeks = 0
# nilai = 0
# while indeks < len(data) :
#     if (data[indeks] == []) :
#         data.remove(data[indeks])
#         dlabel.remove(dlabel[indeks])
#         nilai+=1
#         print(nilai)
#     else :
#         indeks += 1

Ratio = pd.read_pickle('C:/Users/Wellington/Desktop/skripC/processing/FeatureRasioSortircleanser.pkl')
Ratio = Ratio.reset_index()
Rasio = Ratio.drop('index', axis = 1)

vocab = Rasio["Feature"].values.tolist()
print("panjang vocab",len(vocab))
# row =[]
# isi=[]
#
# for b in range(len(vocab)):
#     isi.append(0)
#
# print(data.__len__())
# for a in range(len(data)):
#    row.append(isi)
#
# datarow= np.array(row)#buat dari list jadi array, jadi pecah isinya jadi array
# objek = np.array(data)#melihat data's list untukdipecah jadi array, dimana nilai array sekarang per kata
# tes = pd.DataFrame(datarow,columns = [vocab])
#
# for p_data in range(data.__len__()): #panjang jumlah review (data)>> hitung kebawah
#     print("iterasii -",p_data)
#     for p_kolom in range(vocab.__len__()): #panjang colom, dibawahnya masukkin nilai array review berikutnya (vocab)>> ke kanan
#         for p_objek in range(data[p_data].__len__()) : #panjang masing masing array(len(data[i]))
#             if objek[p_data][p_objek] == vocab[p_kolom] : # cek apakah sama atau tidak antara nilai substring dalam list data[i] dengan kata di kolom itu, kalau sama plus 1
#                 tes.loc[p_data][p_kolom] = tes.loc[p_data][p_kolom] + 1
#
# # tes['Label'] = pickle.load(open(r'C:/Users/Wellington/Desktop/skripC/processing/label.pkl',"rb"))
# tes['Label'] = dlabel.copy()
# tes.to_csv(r'C:/Users/Wellington/Desktop/skripC/processing/dataframe.csv',index = False, header=True)
# tes.to_pickle(r'C:/Users/Wellington/Desktop/skripC/processing/dataframe.pkl')
# # ##############################################################################################################################################
kromosom = 30 #jumlah kromosom dalam populasi
gen = 100 # jumlah gen dalam satu kromosom
population_size = (kromosom,gen)

new_population = np.random.randint(low = 0, high = (len(vocab)-1), size = population_size )

crossover_rate = 0.85 #adalah dari sekian kromosom yang ada idppuplasi, berapa yang berpeluang mengikuti proses crossover, coba uniform 0-1
mutation_rate = 0.1
generation = 100

x_fix =pd.read_pickle(r'C:/Users/Wellington/Desktop/skripC/processing/x_fix_cleanser.pkl')
y_fix = pickle.load(open(r'C:/Users/Wellington/Desktop/skripC/processing/y_fix_cleanser.pkl',"rb"))

x_train, x_test, y_train, y_test = train_test_split(x_fix.to_numpy(), y_fix, test_size=0.20, random_state = None)
atas = 0
print("panjang x_train,x_test,y_train,_y_test berturut turut ",len(x_train), len(x_test), len(y_train), len(y_test))
for a in y_train :
    if a == 1 :
        atas += 1
atas = atas / len(y_train)
bawah = 1 - atas
print("rasio train yes : no ",atas, " : ",bawah)

for a in y_test :
    if a == 1 :
        atas += 1
atas = atas / len(y_test)
bawah = 1 - atas
print("rasio test yes : no ",atas, " : ",bawah)

#database untuk GA
dftrain = pd.DataFrame(x_train, columns = [vocab])
dftrain["Label"] = y_train.copy()
dftest = pd.DataFrame(x_test, columns = [vocab])
dftest["Label"] = y_test.copy()

#inisiasi
bestvalues = []
dfpopulation = pd.DataFrame(new_population)
akurasilist = []
fscoreterpilih = []
akurasiterpilih = []
x = pd.DataFrame(columns=["Akurasi","Fscore","Precision","Recall"])

# # POPULASI AWAL ###################################################################################################################
for i in range(kromosom): #CARI fv POPULASI AWAL
    dffitnessvaluetrain = pd.DataFrame()
    dffitnessvaluetest = pd.DataFrame()
    temporary = pd.DataFrame()
    listindexurut = []
    listnama = []
    # print(dfpopulation.loc[i])
    for j in range(gen) :
        a = dfpopulation.loc[i][j]
        b = vocab[a]
        kata = ""
        kata = kata.join([str(j),". ",b])
        dffitnessvaluetrain[[kata]] = dftrain[[b]].copy()
        dffitnessvaluetest[[kata]] = dftest[[b]].copy()
        listindexurut.append(a)
        listnama.append(b)
    y_test = dftest[['Label']].copy()
    x_test = dffitnessvaluetest
    clf= DecisionTreeClassifier(max_depth = 22)
    y_train = dftrain[['Label']].copy()
    x_train = dffitnessvaluetrain
    y_train = np.ravel(y_train)
    # y_predicted = cross_val_predict(clf, df.to_numpy(), label.to_numpy(), cv=10)
    # print(y_predicted)
    clf = clf.fit(x_train,y_train)
    fitur = clf.feature_importances_
    temporary["importance"] = fitur.copy()
    temporary["feature"] = listindexurut.copy()
    temporary["nama"] = listnama.copy()
    temporary.sort_values("importance", axis = 0, ascending = False,
                inplace = True, na_position ='first')
    temporary = temporary.reset_index()
    for nilai in range(gen) :
        dfpopulation.loc[i][nilai] = temporary["feature"][nilai]
    y_pred = clf.predict(x_test)
    y_latian = clf.predict(x_train)
    # accTest = metrics.accuracy_score(y_test,y_pred)
    CM = metrics.confusion_matrix(y_test, y_pred)
    akurasi = (CM[0][0]+CM[1][1])/ (CM[1][1]+CM[1][0]+CM[0][0]+CM[0][1])
    akurasi = akurasi
    precision = CM[1][1]/(CM[1][1]+CM[0][1])
    recall = CM[1][1]/ (CM[1][1]+CM[1][0])
    fscore = 2 * precision * recall / (precision + recall)
    isi = pd.DataFrame({"Akurasi" : [akurasi],"Fscore" :  [fscore], "Precision" :  [precision],"Recall" : [recall]})
    x = x.append(isi, ignore_index = True)
    akurasilist.append(akurasi)
    akurasiterpilih.append(akurasi)
    fscoreterpilih.append(fscore)
    # fig, ax = plt.subplots(figsize=(50, 25))
    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data,
    #             filled=True, rounded=True,
    #             special_characters=True,feature_names = temporary["nama"],class_names=["0-No" ,"1-Yes"])

    # r = export_text(clf,feature_names = temporary["nama"].tolist(),max_depth = 3)
    # print(" ")
    # print(r)
    # print(" ")
    # r = export_text(clf,feature_names = temporary["nama"].tolist(),max_depth = 24)
    # print(r)
    # temporary.to_csv('C:/Users/Wellington/Desktop/skripC/processing/temporarysebelumcleanser.csv',index = False, header = True)

    # plt.savefig('tree_high_dpi',)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('grafik.png')
    # Image(graph.create_png())
print(tabulate(x, headers='keys', tablefmt='psql'))
bestgen = max(akurasilist)
for indeks in range(len(akurasilist)) :
    if akurasilist[indeks] == bestgen :
        bestakurasi = akurasiterpilih[indeks]
        bestfscore = fscoreterpilih[indeks]
bestvalues.append(bestgen)
dfpopulation["akurasi"] = akurasilist.copy()
dfpopulation.sort_values("akurasi", axis = 0, ascending = False,
                inplace = True, na_position ='first')
dfpopulation = dfpopulation.reset_index()
dfpopulation = dfpopulation.drop('index', axis = 1)
dfpopulation = dfpopulation.drop('akurasi', axis = 1)
awalakurasi = bestakurasi
awalfscore = bestfscore
awal = bestgen

start = dt.now()
print("waktu mulai : ",str(start))
print("---------------------------------------------------")
################################################################
# MULAI PROSES GENETIC ALGORITHM####################################################################################################################3
for generasi in range(generation) : #iterasi sampai generasi n
    print("generasi ke-",generasi+1)
    akurasilist =[]
    akurasiterpilih = []
    fscoreterpilih = []
    offspring = []
    matingpool = []
    tabelfv = pd.DataFrame(columns=["Akurasi","Fscore","Precision","Recall"])
    tabelfvtemp = pd.DataFrame(columns=["Akurasi","Fscore","Precision","Recall"])
    for apa in range(len(dfpopulation)) : #acak dari random uniform, untuk dptin mating pool
        acakparent = np.random.uniform()
        if acakparent < crossover_rate :
            matingpool.append(dfpopulation.loc[apa])
    for index_parent in range(len(matingpool)-1): #crossover
        parent1 = matingpool[index_parent].copy()
        parent2 = matingpool[index_parent+1].copy()
        number_uniform_crossover = int(gen/2)
        listindexrandom = []
        crossover = []
        for CO in range(number_uniform_crossover) :
            crossover.append(parent1[CO])
            crossover.append(parent2[CO])
        # random.shuffle(crossover)
        offspring.append(crossover)
    #mutation
    listindexmutation=[]
    for indeks in range(int(mutation_rate*gen)): #buat list index yang akan dimutasi
        randomu = np.random.randint(low = 0 , high = gen-1)
        if randomu not in listindexmutation:
            listindexmutation.append(randomu)
        else :
            while randomu in listindexmutation :
                randomu = np.random.randint(low = 0 , high = gen-1)
                if randomu not in listindexmutation :
                    listindexmutation.append(randomu)
                    break

    for index_parent in range(len(offspring)): #mutation
        mutation = offspring[index_parent]
        number_mutation = int(len(listindexmutation))
        listindexrandom = []
        for a in range(number_mutation) :
            b = listindexmutation[a]
            if (len(vocab)-1-mutation[b]) not in mutation : #kalau di index mau diganti vocab - isi index, dan trnyata blm ada isinya di dlm indexnya, yaudah sikat
                mutation[b] = len(vocab)- 1 - mutation[b]
            else :
                mutate = np.random.randint(low = 0 , high = len(vocab)-1)
                if mutate not in mutation :
                    # b = listindexmutation[a]
                    mutation[b] = mutate
                else :
                    while mutate in mutation :
                        mutate = np.random.randint(low = 0 , high = len(vocab)-1)
                        if mutate not in mutation :
                            # b = listindexmutation[a]
                            mutation[b] = mutate
                            break
    # print(dfpopulation) #tunjukkan hasil mutation
    selectionpool = pd.DataFrame()
    selectionpool = dfpopulation.copy() #gabungin offspring hasil crossover dengan parent di mating pool
    if (offspring != []) :
        selectionpool = selectionpool.append(offspring)
    selectionpool = selectionpool.reset_index()
    selectionpool = selectionpool.drop('index', axis = 1)
    for i in range(len(selectionpool)):
        # fitness_value = fitnessvalue(new_population)
        dffitnessvaluetrain = pd.DataFrame()
        dffitnessvaluetest = pd.DataFrame()
        temporary = pd.DataFrame()
        listindexurut = []
        for j in range(gen) :
            a = int(selectionpool.loc[i][j])
            b = vocab[a]
            kata = ""
            kata = kata.join([str(j),". ",b])
            dffitnessvaluetrain[[kata]] = dftrain[[b]].copy()
            dffitnessvaluetest[[kata]] = dftest[[b]].copy()
            listindexurut.append(a)
        x_test = dffitnessvaluetest #besok dicari cara
        y_test = dftest[['Label']].copy()
        #besok dicari cara
        clf= DecisionTreeClassifier(max_depth = 22)  #seberapa jauh kamu mau mencabangkan pohon
        x_train = dffitnessvaluetrain
        y_train = dftrain[['Label']].copy()
        y_train = np.ravel(y_train)
        clf = clf.fit(x_train,y_train)
        fitur = clf.feature_importances_
        temporary["importance"] = fitur.copy()
        temporary["feature"] = listindexurut.copy()
        temporary.sort_values("importance", axis = 0, ascending = False,
                    inplace = True, na_position ='first')
        temporary = temporary.reset_index()
        for nilai in range(gen) :
            selectionpool.loc[i][nilai] = temporary["feature"][nilai]
        y_pred = clf.predict(x_test)
        CM = metrics.confusion_matrix(y_test, y_pred)
        akurasi = (CM[0][0]+CM[1][1])/ (CM[1][1]+CM[1][0]+CM[0][0]+CM[0][1])
        precision = CM[1][1]/(CM[1][1]+CM[0][1])
        recall = CM[1][1]/ (CM[1][1]+CM[1][0])
        fscore = 2 * precision * recall / (precision + recall)
        isi = pd.DataFrame({"Akurasi" : [akurasi],"Fscore" :  [fscore], "Precision" :  [precision],"Recall" : [recall]})
        tabelfvtemp = tabelfvtemp.append(isi, ignore_index = True)
        akurasi = akurasi
        akurasilist.append(akurasi)
        akurasiterpilih.append(akurasi)
        fscoreterpilih.append(fscore)
    selectionpool["akurasi"] = akurasilist.copy()
    selectionpool.sort_values("akurasi", axis = 0, ascending = False,
                     inplace = True, na_position ='first')
    selectionpool = selectionpool.reset_index()
    selectionpool = selectionpool.drop('index', axis = 1)
    selectionpool = selectionpool.drop('akurasi', axis = 1)

    tabelfvtemp.sort_values("Akurasi", axis = 0, ascending = False,
                     inplace = True, na_position ='first')
    tabelfvtemp = tabelfvtemp.reset_index()
    tabelfvtemp = tabelfvtemp.drop('index', axis = 1)

    listfv = pd.DataFrame()
    listfv["akurasilist"] = akurasilist.copy()
    listfv["akurasiterpilih"] = akurasiterpilih.copy()
    listfv["fscoreterpilih"] = fscoreterpilih.copy()
    listfv.sort_values("akurasilist", axis = 0, ascending = False,
                     inplace = True, na_position ='first')

    akurasilist = listfv["akurasilist"].values.tolist()
    akurasiterpilih = listfv["akurasiterpilih"].values.tolist()
    fscoreterpilih = listfv["fscoreterpilih"].values.tolist()
    rangeroulette = sum(akurasilist)
    hasilroulette = []
    selectedpopulation = pd.DataFrame()
    selectedfitness = []
    for i in range(len(selectionpool)) : #pembuatan roulette wheel
        # print("rol")
        if i == 0 :
            hasilroulette.append(np.arange(0,akurasilist[0],0.001))
        else :
            hasilroulette.append(np.arange(akurasilist[i-1],akurasilist[i],0.001))
    roulette = 0
    akurasifix = []
    fscorefix = []
    while roulette < kromosom : #selection
        angka = np.random.uniform(low = 0, high = rangeroulette)
        angka = round(angka,3)
        b = 0
        while b < (len(akurasilist)) :
            if angka in hasilroulette[b]:
                selectedpopulation = selectedpopulation.append(selectionpool.loc[b],ignore_index=True)
                selectionpool = selectionpool.drop([b])
                selectionpool = selectionpool.reset_index()
                selectionpool = selectionpool.drop('index', axis = 1)
                selectedfitness.append(akurasilist[b])
                akurasifix.append(akurasiterpilih[b])
                fscorefix.append(fscoreterpilih[b])
                akurasilist.remove(akurasilist[b])
                akurasiterpilih.remove(akurasiterpilih[b])
                fscoreterpilih.remove(fscoreterpilih[b])
                tabelfv = tabelfv.append(tabelfvtemp.loc[b],ignore_index = True)
                tabelfvtemp = tabelfvtemp.drop([b])
                tabelfvtemp = tabelfvtemp.reset_index()
                tabelfvtemp = tabelfvtemp.drop('index', axis = 1)
                roulette += 1
                break
            else :
                b+= 1
    dfpopulation = selectedpopulation.copy()
    # for p in range(kromosom) :
    #     print("generasi-",generasi+1," di kromosom ", p+1,"punya accuracy: ",selectedfitness[p])
    print(tabulate(tabelfv, headers='keys', tablefmt='psql'))
    bestgen = max(selectedfitness)
    for i in range(len(selectedfitness)) :
        if bestgen == selectedfitness[i]:
            terbaik = selectedpopulation.loc[i]
            akhirakurasi = akurasifix[i]
            akhirfscore = fscorefix[i]
    bestvalues.append(bestgen)
# plt.plot(bestvalues)
# bad = "rt. mutasi : ",mutation_rate,", rt. crossover : ",crossover_rate,", populasi : ",kromosom,", generasi : ",generation
# plt.title(bad)
# plt.xlabel("Iteration")
# plt.ylabel("Fitness Value : ")
# plt.show()
# print(" Yang Terbaik : ")
# for j in range(len(terbaik)) :
#     index = int(terbaik[j])
#     print(vocab[index], end = " ")
end = dt.now()
print("waktu selesai : ",str(end))
print("------------------------------------------------------------")
total = end-start
print("total waktu : ",str(total))
print("-------------------------------------------------------------")

akhir = bestgen

print("besar kenaikan fitness value dari ",(awal)," - ",(akhir)," adalah : ",(akhir - awal))
print("besar kenaikan akurasi dari ",(awalakurasi)," - ",(akhirakurasi)," adalah : ",(akhirakurasi - awalakurasi))
print("besar kenaikan f-score dari ",(awalfscore)," - ",(akhirfscore)," adalah : ",(akhirfscore - awalfscore))

print("---------------selesai--------------------------------")


listindexurut = []
listindex = []
dffitnessvaluetrain = pd.DataFrame()
dffitnessvaluetest = pd.DataFrame()
temporary = pd.DataFrame()
for j in range(len(terbaik)) :
    a = int(terbaik[j])
    b = vocab[a]
    kata = ""
    kata = kata.join([str(j),". ",b])
    dffitnessvaluetrain[[kata]] = dftrain[[b]].copy()
    dffitnessvaluetest[[kata]] = dftest[[b]].copy()
    listindexurut.append(b)
    listindex.append(a)
x_test = dffitnessvaluetest
y_test = dftest[['Label']].copy()
x_train = dffitnessvaluetrain
y_train = dftrain[['Label']].copy()
# y_train = np.ravel(y_train)
clf= DecisionTreeClassifier(max_depth = 22)
clf = clf.fit(x_train,y_train)
fitur = clf.feature_importances_
temporary["importance"] = fitur.copy()
temporary["feature"] = listindexurut.copy()
temporary["indeks dalam vocab"] = listindex.copy()
temporary.sort_values("importance", axis = 0, ascending = False,
            inplace = True, na_position ='first')
temporary = temporary.reset_index(drop = True)
temporary.to_csv('C:/Users/Wellington/Desktop/skripC/processing/temporarysesudahcleanser.csv',index = False, header = True)
r = export_text(clf,feature_names = temporary["feature"].tolist(),max_depth = 3)
print(" ")
print(r)
print(" ")
r = export_text(clf,feature_names = temporary["feature"].tolist(),max_depth = 24)
print(r)

# tree.plot_tree(clf,feature_names =[temporary["feature"]] ,class_names=["0-No" ,"1-Yes"])
