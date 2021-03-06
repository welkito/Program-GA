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
from nltk import tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

datadir = "C:/Users/Wellington/Desktop/skripC/processing/"
dfdata = pd.read_csv(datadir+"dataset.csv")

# dibawah ini process sortir data awal biar jadi dataframe yang sesungguhnya
#######################################################################################
panjangawal = dfdata.__len__()

stop_factory = StopWordRemoverFactory().get_stop_words()
listStopwordBaru = open(datadir+"kata3.csv", 'r')
reader = csv.reader(listStopwordBaru)
for stopwords1 in reader:
    katastop = stop_factory+stopwords1
stop_inggris = set(stopwords.words('english'))

#rumus untuk hilangin tanda baca
def tanpatanda(text):
    textbersih = "".join([ce for ce in text if ce not in string.punctuation])
    return textbersih
dfdata['Clean'] =  dfdata['TextReview'].apply(lambda x: x.replace(',', ''))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda y: y.replace('u\'' ,  ''))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('Trusted Review\'' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('.' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('!' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('&' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('?' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('[' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace(']' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('u"' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('\ ' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('ud83d' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('u2661' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: z.replace('u201d' , ' '))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda z: tanpatanda(z))
dfdata['Clean'] =  dfdata['Clean'].apply(lambda x: re.sub(r'[0-9]',r'',x))

print(dfdata['Clean'])

#penyortiran dari nilai maybe, tidak ada keputusan repurchase, dan data dengan text review kosong
dfdata['Purchase'] = dfdata['RatingPerCategory'].str.find(' Repurchase?Maybe')
print(dfdata)
dfdata = dfdata[(dfdata.Purchase == -1)]
dfdata['Tipe'] = dfdata['RatingPerCategory'].str.find(' Repurchase?')
dfdata= dfdata[(dfdata.Tipe != -1)]
dfdata['Bersih'] = dfdata['TextReview'].str.find(" ")
dfdata = dfdata[dfdata.Bersih != -1]
dfdata['asd'] = dfdata['productCategory'].str.find("treatment")
dfdata = dfdata[dfdata.asd == 0]
dfdata = dfdata.drop('asd', axis = 1)
dfdata = dfdata.reset_index()
dfdata = dfdata.drop('index', axis = 1)

dfdataLabel = dfdata[['DictioPerCategory','Clean']].copy()
print(dfdataLabel)
dfdataLabel['Repurchase'] = dfdataLabel['DictioPerCategory'].str.find('No')
dfdataLabel['Label']= dfdataLabel['Repurchase'].copy()
dfdataLabel.loc[dfdataLabel['Repurchase'] == -1, ['Label']] = 1
dfdataLabel.loc[dfdataLabel['Repurchase'] != -1, ['Label']] = 0
#
# #tekken-ize, maksudnya tokenize
print("banyak review setelah sortir",len(dfdataLabel))
dfdataLabel['CategoryToken'] = dfdataLabel['DictioPerCategory'].apply(lambda s: nltk.word_tokenize(s))
dfdataLabel['wordToken'] = dfdataLabel['Clean'].apply(lambda a: nltk.word_tokenize(a))
listbaru = dfdataLabel['wordToken'].values.tolist()

##untuk ubah kata bagusss jadi bagus dan hilangi kata yang nilainya 2 atau 1 huruf
listdata=[]
data=[]
for kalimat in listbaru:
    listdata=[]e
    for kata in kalimat :
        if len(kata) < 3:
            kata = ""
        temporary = []
        for h in kata :
            temporary.append(h)
        statusawal= True
        statusakhir = True
        counter =0
        for huruf in range(len(temporary)-1):
            if temporary[huruf].lower() == temporary[huruf+1].lower():
                counter +=1
                statusawal= True
                statusakhir = True
            if temporary[huruf].lower() != temporary[huruf+1].lower():
                statusawal = False
            if counter == 2 :
                mulai = huruf
            if huruf == (len(temporary) - 2) :
                statusakhir = False
            if (counter > 1 and statusawal == False) :
                counter = 0
                for angka in range(mulai,huruf+1):
                    temporary[angka] = ""
            if (counter > 1 and statusakhir == False) :
                counter = 0
                for angka in range(mulai,huruf+2):
                    temporary[angka] = ""
        listdata.append("".join(temporary))
    data.append(listdata)

# penghilangan stopword dan kata sambung
print(len(data))
listbaru = data.copy()
data =[]
indes = 0;
for kalimat in listbaru:
    listdata =[]
    hasildetoken = []
    # print('iterasi',indes)
    indes += 1
    textbersih = []
    for ce in kalimat :
        ab = ce.lower()
        if ab not in katastop :
            if ab not in stop_inggris:
                listdata.append(stemmer.stem(ab))
    hasildetoken = TreebankWordDetokenizer().detokenize(listdata)
    data.append(hasildetoken)
##review bersih dari stopwords, kata typo panjang, kata yang panjangnya kurang dari 3, kata berimbuhan, angka dan tanda baca.
hasilakhir =[]
for i in range(len(data)) :
    hasilakhir.append(nltk.word_tokenize(data[i]))

dfdataLabel['FIXBERSIH'] = hasilakhir
dfdataLabel = dfdataLabel.reset_index()
dfdataLabel = dfdataLabel.drop('index', axis = 1)
dfdataLabel.to_pickle('C:/Users/Wellington/Desktop/skripC/processing/dfdatalabel.pkl')
# selesai untuk penyortiran dataframem awal dari kata tidak penting
# # ###################################################################################################################################
#pecah 80/20 sesuai prinsip pareto
dfdataLabel = pd.read_pickle(datadir+"dfdatalabelcleanser.pkl")

msk = np.random.rand(len(dfdataLabel)) < 0.8
train = dfdataLabel[msk]
test = dfdataLabel[~msk]
train.to_pickle('C:/Users/Wellington/Desktop/skripC/processing/train.pkl')
test.to_pickle('C:/Users/Wellington/Desktop/skripC/processing/test.pkl')
train = pd.read_pickle(datadir+"traincleanser.pkl")
test = pd.read_pickle(datadir+"testcleanser.pkl")
print("panjang test : ",len(test))
print("panjang train : ",len(train))
atastrain = train[train.Label == 1]
atas = len(atastrain) / len(train)
bawah = 1 - atas
print("rasio  train(yes : no) : ",atas, " : ",bawah)
atastest = test[test.Label == 1]
atassatu = len(atastest) / len(test)
bawahtest = 1- atassatu
print("rasio tes ( yes : no) : ",atassatu," : ",bawahtest)
#
#########################################################################################################################################################
# MULAI DARI SINI UDA PAKE TRAIN SET
# membuat label list untuk dataset
train = pd.read_pickle(datadir+"train.pkl")
train = train.reset_index()
train = train.drop('index', axis = 1)

data = train['FIXBERSIH'].values.tolist()
dlabel = []
for kiri,kanan in train['Label'].items() :
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

data_flat= [stemmer.stem(token.lower())for sent in data for token in sent]
fdist = FreqDist(data_flat)
vocab = []
numWord = 0
for key,val in fdist.items():
    vocab.append(str(key))
    numWord = numWord + val

print("Panjang vocab treatment sebelum di buat rasio", len(vocab))

# pembuatan ratio indexing untuk semua feature

RatioFeature = []
frekuensi = []
for indexfeature in range(len(vocab)) : #pengulangan ganti index untuk dicek lagi smua reviewnya
    print("iterasi ke-",indexfeature)
    counter= 0
    jumlah = 0
    PecahanRev = 0
    jumlahangka = 0
    for indexreview in range(data.__len__()):
        atasRev = 0
        jumlah = False
        for jumlahkataperreview in range(data[indexreview].__len__()): # pengulangan untuk satu review
            if vocab[indexfeature].lower() == data[indexreview][jumlahkataperreview].lower() :
                atasRev = atasRev +1
                jumlah = True
        pecahan = atasRev/data[indexreview].__len__()
        if pecahan > 0 :
            counter += 1
        if dlabel[indexreview] == 0 :
            pecahan = pecahan * -1
        if jumlah == True :
            jumlahangka += 1
        PecahanRev = PecahanRev + pecahan
    PecahanRev = PecahanRev / counter
    RatioFeature.append(str(PecahanRev))
    frekuensi.append(jumlahangka)

Ratio = pd.DataFrame(RatioFeature,columns = ["Rasio"])
mutlak = [float(ele) for ele in RatioFeature]
mutlaks = [abs(ele) for ele in mutlak]
Ratio["RasioMutlak"] = mutlaks.copy()
Ratio["Feature"] = vocab.copy()
Ratio["FrekuensiKemunculanTiapDokumen"] = frekuensi.copy()


Ratio.sort_values("RasioMutlak", axis = 0, ascending = False,
                 inplace = True, na_position ='first')
Ratio.to_pickle(r'C:/Users/Wellington/Desktop/skripC/processing/FeatureRasi.pkl')
Ratio.to_csv('C:/Users/Wellington/Desktop/skripC/processing/FeatureRasi.csv',index = False, header = True)

# HASIL RATIO YANG UDA DIURUTIN DAN UDA DILETAKKAN DI FEATURE RASIO
Ratio = pd.read_pickle((datadir+"FeatureRasi.pkl"))
indeks = 0
Rasio = Ratio.reset_index()
Rasio = Rasio.drop('index', axis = 1)
for indeks in range(len(Rasio)) :
    if Rasio["FrekuensiKemunculanTiapDokumen"][indeks] <= 5 :
        Rasio = Rasio.drop([indeks])
        print("iterasi ilang ke-",indeks)
Rasio.to_pickle(r'C:/Users/Wellington/Desktop/skripC/processing/FeatureRasioSorti.pkl')
Rasio.to_csv('C:/Users/Wellington/Desktop/skripC/processing/FeatureRasioSortir.csv',index = False, header = True)
print("panjang vocab  skrg ",Rasio)
####################################################################################################################

# pickle.dump(data,open(r'C:/Users/Wellington/Desktop/skripC/processing/data.pkl',"wb"))
# pickle.dump(dlabel,open(r'C:/Users/Wellington/Desktop/skripC/processing/label.pkl',"wb"))
