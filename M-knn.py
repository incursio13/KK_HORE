# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
import operator
import time

def majority_vote(knn, labels):
    knn = [k[0, 0] for k in knn]    
    dictionary = {}
    for idx in knn:
        if labels[idx] in dictionary.keys():
            dictionary[labels[idx]] = dictionary[labels[idx]] + 1
        else:
            dictionary[labels[idx]] = 1
    #print a
    result = sorted(dictionary.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]
#    hasil= sorted(dictionary.iteritems(), key=operator.itemgetter(1), reverse=True)[0]
#    hasil1= sorted(dictionary.iteritems(), key=operator.itemgetter(1), reverse=True)
#    print hasil
#    print hasil1
#    print result
#    print dictionary
    return result

def doWork(train, test, labels_train, labels_test):
    
    for i in range(2,9):        
        #k = int(raw_input("Jumlah K : "))
        k=i
        train_mat = np.mat(train)
        result=[]
        start = time.time()
        for test_sample in test:
            #hitung knn,lalu diurutkan
            knn = np.argsort(np.sum(np.power(np.subtract(train_mat, test_sample), 2), axis=1), axis=0)[:k]
            #hitung banyak tetangga terdekat berdasarkan nilai k
            prediction = majority_vote(knn, labels_train)
            result.append(prediction)      
        #hitung akurasi
        correct = 0        
        for i in range(len(test)-1):
            if labels_test[i] == result[i]:
                correct += 1
            #print "kelas : "+labels_test[i]+"   hasil :"+result[i]
        print "k        : " + str(k)
        print "Akurasi  : " + str((correct/float(len(test))) * 100.0) + " % " 
        print "Run time : " + str(time.time()-start)

def splitDataset(data, splitRatio):
    split=random.sample(data.index, (int)(splitRatio*len(data)))
    train= data.ix[split]
    test= data.drop(split)
    train.sort_index(inplace=True)
    test.sort_index(inplace=True)
    return [train, test]

if __name__ == '__main__':
    dataset=raw_input("Masukkan nama data train : ");
    datatest=raw_input("Masukkan nama data test : ");
    try :
        train = pd.read_csv(dataset)
        test = pd.read_csv(datatest)

        #cek data kategorikal
        for i in range(len(train.ix[0])-1):
            column_name= train.columns.values[i]
            if train[column_name].dtype==object:
                train[column_name]=train[column_name].astype('category')
                train[column_name]=train[column_name].cat.codes
    
        for i in range(len(test.ix[0])-1):
            column_name= test.columns.values[i]
            if test[column_name].dtype==object:
                test[column_name]=test[column_name].astype('category')
                test[column_name]=test[column_name].cat.codes

        #digunakan jika cuma 1 data set
        #splitRatio =float(raw_input("Split ratio (0-1): "));
        #train, test= splitDataset(data, splitRatio)    
        
        # memisahkan label dari data train
        labels_train = []
        for a in train.index:
            labels_train.append(train.ix[a][len(train.columns)-1])        
        train_fix=train.drop(train.columns[len(train.columns)-1],axis=1)
        
        # memisahkan label dari data test
        labels_test = []
        for a in test.index:
            labels_test.append(test.ix[a][len(test.columns)-1])        
        test_fix=test.drop(test.columns[len(test.columns)-1],axis=1)
        
        train_fix = train_fix.as_matrix()
        test_fix = test_fix.as_matrix()
        
        doWork(train_fix, test_fix, labels_train, labels_test)
        
    except IOError as e:
        print "Tidak ditemukan file"
    except ValueError:
        print "Ratio salah"