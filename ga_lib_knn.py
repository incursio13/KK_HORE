# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 22:49:50 2016

@author: incr
"""
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import random

def kromosom():
    population=k
    return [random.randint(1,125) for x in range(population)]
    
def binary(parent, length):
    biner=[]
    length="{0:0b}".format(length)
    length=len(length)
    for x in range(len(parent)):
        temp="{0:0b}".format(x)
        biner.append(temp.zfill(length))
    return biner
    
def desimal(biner):
    return [int(x,2) for x in biner]

def splitDataset(data, splitRatio):
    split=random.sample(data.index, (int)(splitRatio*len(data)))
    train= data.ix[split]
    test= data.drop(split)
    train.sort_index(inplace=True)
    test.sort_index(inplace=True)
    return [train, test]

def fitness(x):
    nbrs = KNeighborsClassifier(n_neighbors=x, algorithm='auto').fit(train_fix,labels_train)
    y_pred=nbrs.predict(test_fix)
    
    correct=0
    for i in range(len(labels_test)):
        if labels_test[i] == y_pred[i]:
            correct += 1
    return correct/float(len(labels_test))

def cumulatived():
    fit=[]
    for x in Krom:
        fit.append(fitness(x))
    
    cumulative=[]
    prob=[]    
    for x in range(len(Krom)):
        prob.append(fit[x]/sum(fit))
        if x!=0:
            cumulative.append(prob[x]+cumulative[x-1])        
        else:
            cumulative.append(prob[x])        
    return prob, cumulative

def rouletteWheel():
    roulette=[]
    Ran=[]
    for x in range(len(cumulative)):
        R=random.random()    
        Ran.append(R)
        for i in range(len(cumulative)):
            if R<cumulative[i]:
               roulette.append(Krom[i])     
               break
    
    return roulette, Ran

def crossover(length):
    R2=0
    parent=[]
    for x in range(int(k*0.8)):
        R1=random.randint(0,k-1)
        while R1==R2:
            R1=random.randint(0,k-1)
        parent.append(Krom[R1])
        R2=R1
    bins=binary(parent, length)

    child=[]    
    for x in range(len(parent)):
        if x%2==1:
            continue
        R1=random.randint(1,len(bins[x])-1)
#        R1=len(bins[x])/2
        male=bins[x][:R1]
        female=bins[x+1][R1:]
        temp1=int(male,2)
        temp2=int(female,2)
        while temp1==0 and temp2==0:
            R1=random.randint(0,len(bins[x])-1)
            male=bins[x][:R1]
            female=bins[x+1][R1:]
            temp1=int(male,2)
            temp2=int(female,2)
        if desimal(male+female)<length:
            child.append(male+female)
        if desimal(female+female)<length:
            child.append(female+male)
        
    return child

def fixing_dataset(train):
    # memisahkan label dari data train
    if jumlah_dataset==2:
        splitRatio=0.83333334
        train, test= splitDataset(train, splitRatio)    
    
    labels_train = []
    for a in train.index:
        labels_train.append(train.ix[a][len(train.columns)-1])        
    train_fix=train.drop(train.columns[len(train.columns)-1],axis=1)
    
    # memisahkan label dari data test
    labels_test = []
    for a in test.index:
        labels_test.append(test.ix[a][len(test.columns)-1])        
    test_fix=test.drop(test.columns[len(test.columns)-1],axis=1)
    
    train_fix = np.mat(train_fix)
    test_fix = np.mat(test_fix)
    
    return train_fix, test_fix, labels_train, labels_test

if __name__ == '__main__':
    jumlah_dataset=int(raw_input("Jumlah dataset : "))
    dataset=raw_input("Masukkan nama data train : ")
#    dataset="train_norm.csv"
#    datatest="test_norm.csv"
#    dataset="iris.csv"
    if jumlah_dataset==2:
        datatest=raw_input("Masukkan nama data test : ");
    try :
        train = pd.read_csv(dataset)
        if jumlah_dataset==2:
            test = pd.read_csv(datatest)
    
        for i in range(len(train.ix[0])-1):
            column_name= train.columns.values[i]
            if train[column_name].dtype==object:
                train[column_name]=train[column_name].astype('category')
                train[column_name]=train[column_name].cat.codes
                
        if jumlah_dataset==2:        
            for i in range(len(test.ix[0])-1):
                column_name= test.columns.values[i]
                if test[column_name].dtype==object:
                    test[column_name]=test[column_name].astype('category')
                    test[column_name]=test[column_name].cat.codes 
        
        train_fix, test_fix, labels_train, labels_test= fixing_dataset(train)
    
        k=10
        Krom=kromosom()
    
        new_child=[]
        for x in range(1,201):
            print "loop = "+str(x)
            Krom=Krom[:k]+new_child
            prob_fitness, cumulative=cumulatived()
            roulette, Random_log=rouletteWheel()        
            Krom=roulette
            cross=crossover(len(labels_train), )
            new_child=desimal(cross)
    
        k_fixed=prob_fitness.index(max(prob_fitness))
        print "k        : " + str(Krom[k_fixed])
        correct = fitness(Krom[k_fixed])
        print "Akurasi  : " + str(correct * 100.0) + " % "
#        Krom=Krom[:k]+new_child
#        for x in Krom:     
#            print "k        : " + str(x)
#            correct = fitness(x)
#            print "Akurasi  : " + str(correct * 100.0) + " % " 
        
    except IOError as e:
        print "Tidak ditemukan file"
    except ValueError:
        print "Ratio salah"