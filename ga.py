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

if __name__ == '__main__':
    dataset=raw_input("Masukkan nama data train : ");
#    datatest=raw_input("Masukkan nama data test : ");
    try :
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
        splitRatio =float(raw_input("Split ratio (0-1): "));
        train, test= splitDataset(train, splitRatio)    
        
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
        
    except IOError as e:
        print "Tidak ditemukan file"
    except ValueError:
        print "Ratio salah"