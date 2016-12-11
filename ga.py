import numpy as np
import pandas as pd
import random
import operator
import time


def splitDataset(data, splitRatio):
    split=random.sample(data.index, (int)(splitRatio*len(data)))
    train= data.ix[split]
    test= data.drop(split)
    train.sort_index(inplace=True)
    test.sort_index(inplace=True)
    return [train, test]

def kromosom(data, k):
    population=k
    return [random.randint(1,125) for x in range(population)]

def similarity(result, labels, idx_sample):
    result = [k[0, 0] for k in result] 
    sim_point=0
    
    for x in result:
        if labels[x]==labels[idx_sample]:
            sim_point+=1
    #print sim_point
    return sim_point

def validity(train, labels_train, k, krom):
    fitness=[]
    for x in krom:
        train_mat = np.mat(train)
        count=0
        valid=[]
        for train_sample in train:
            result=np.argsort(np.sqrt(np.sum(np.power(np.subtract(train_mat, train_sample), 2), axis=1)), axis=0)[1:(x+1)]
            similar=float(similarity(result, labels_train, count))/x
            valid.append(similar)
            #print similar
            count+=1
        fitness.append(valid)
    return fitness

def rouletteWheel():
    
    return None
    
def binary(K):
    return ["{0:07b}".format(x) for x in K]
    
def desimal(K):
    return [int(K,2) for x in K]

if __name__ == '__main__':
#    dataset=raw_input("Masukkan nama data train : ")
    dataset="iris.csv"
#    datatest=raw_input("Masukkan nama data test : ");
  #  try :
    train = pd.read_csv(dataset, header=None)
#        test = pd.read_csv(datatest, header=None)

    #cek data kategorikal
    for i in range(len(train.ix[0])-1):
        column_name= train.columns.values[i]
        if train[column_name].dtype==object:
            train[column_name]=train[column_name].astype('category')
            train[column_name]=train[column_name].cat.codes

#        for i in range(len(test.ix[0])-1):
#            column_name= test.columns.values[i]
#            if test[column_name].dtype==object:
#                test[column_name]=test[column_name].astype('category')
#                test[column_name]=test[column_name].cat.codes

    #digunakan jika cuma 1 data set
    #splitRatio =float(raw_input("Split ratio (0-1): "));
    splitRatio=0.83333334
    train, test= splitDataset(train, splitRatio)    
    ""
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
  
#        doWork(train_fix, test_fix, labels_train, labels_test)
    #GA start
    
    k=3
    Krom=kromosom(train, k)
    binary_K=binary(Krom)
    
    fitness=validity(train_fix,labels_train, k, Krom)
        
        
#    except IOError as e:
#        print "Tidak ditemukan file"
#    except ValueError:
#        print "Ratio salah"