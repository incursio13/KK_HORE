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
    return result

def weight(knn, validity, labels):
    knn = [k[0, 0] for k in knn] 
    W=[]
    for x in range(len(knn)):
        W.append(validity[x]*1/(knn[x]+0.5))
    return labels[np.array(W).argsort()[::-1][:1]]

def doWork(train, test, labels_train, labels_test, validity):
    train_mat = np.mat(train)
    result=[]
    start = time.time()
    for test_sample in test:
        #hitung knn,lalu diurutkan, knn biasa 
        knn = np.sqrt(np.sum(np.power(np.subtract(train_mat, test_sample), 2), axis=1))
        
        #pembobotan
        prediction= weight(knn, validity, labels_train)
        result.append(prediction)      

    correct = 0        
    for i in range(len(test)-1):
        if labels_test[i] == result[i]:
            correct += 1

    print "Akurasi  : " + str((correct/float(len(test))) * 100.0) + " % " 
    print "Run time : " + str(time.time()-start)

def splitDataset(data, splitRatio):
    split=random.sample(data.index, (int)(splitRatio*len(data)))
    train= data.ix[split]
    test= data.drop(split)
    train.sort_index(inplace=True)
    test.sort_index(inplace=True)
   # print len(data)
#    train=data[:int(len(data)*splitRatio)]
#    test=data[int(len(data)*splitRatio)+1:]
    return [train, test]

def similarity(result, labels, idx_sample):
    result = [k[0, 0] for k in result] 
    sim_point=0
    
    for x in result:
        if labels[x]==labels[idx_sample]:
            sim_point+=1
    #print sim_point
    return sim_point

def validity(train, labels_train, k):
    train_mat = np.mat(train)
    print "k        : " + str(k)
    count=0
    valid=[]
    for train_sample in train:
        result=np.argsort(np.sqrt(np.sum(np.power(np.subtract(train_mat, train_sample), 2), axis=1)), axis=0)[1:(k+1)]
        similar=float(similarity(result, labels_train, count))/k
        valid.append(similar)
        #print similar
        count+=1
    
    return valid

if __name__ == '__main__':
#    dataset=raw_input("Masukkan nama data train : ");
#    datatest=raw_input("Masukkan nama data test : ");
    dataset="train_norm.csv"
    datatest="test_norm.csv"
    try :
        train = pd.read_csv("train_norm.csv")
        test = pd.read_csv("test_norm.csv")

        #cek data kategorikal
        for i in range(len(train.ix[0])-1):
            column_name= train.columns.values[i]
            if train[column_name].dtype==object:
                train[column_name]=train[column_name].astype('category')
                train[column_name]=train[column_name].cat.codes
            
                
        #2 data set
        for i in range(len(test.ix[0])-1):
            column_name= test.columns.values[i]
            if test[column_name].dtype==object:
                test[column_name]=test[column_name].astype('category')
                test[column_name]=test[column_name].cat.codes

        #digunakan jika cuma 1 data set
        #splitRatio =float(raw_input("Split ratio (0-1): "));
#        splitRatio=0.8333333334
#        train, test= splitDataset(train, splitRatio)    
        
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
        
        k=5
#        a=euclid_train(train_fix,labels_train, k)
        for k in range (95,99):
            validity_fix=validity(train_fix,labels_train, k)
            doWork(train_fix, test_fix, labels_train, labels_test, validity_fix)
#        validity_fix=0
#        doWork(train_fix, test_fix, labels_train, labels_test, validity_fix)
        
    except IOError as e:
        print "Tidak ditemukan file"
    except ValueError:
        print "Ratio salah"