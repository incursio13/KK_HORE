import numpy as np
import pandas as pd
import random
    
def weight(knn, validity):
    knn = [k[0, 0] for k in knn] 
    W=[]
    for x in range(len(knn)):
        W.append(validity[x]*1/(knn[x]+0.5))

    return labels_train[np.array(W).argsort()[::-1][:1]]


def doWork(validity):
    result=[]
    for test_sample in test_fix:
        knn = np.sqrt(np.sum(np.power(np.subtract(train_fix, test_sample), 2), axis=1))           
        #pembobotan
        prediction= weight(knn, validity,)
        result.append(prediction)      
    #hitung akurasi
    correct = 0        
    for i in range(len(test)-1):
        if labels_test[i] == result[i]:
            correct += 1
            
    return correct/float(len(labels_test)) 

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

def kromosom():
    population=k
    return [random.randint(1,125) for x in range(population)]

def similarity(result, idx_sample):
    result = [k[0, 0] for k in result] 
    sim_point=0
    
    for x in result:
        if labels_train[x]==labels_train[idx_sample]:
            sim_point+=1
    #print sim_point
    return sim_point

def validity(x):
    valid=[]
    count=0
    for train_sample in train_fix:
        result=np.argsort(np.sum(np.power(np.subtract(train_fix, train_sample), 2), axis=1), axis=0)[1:(x+1)]
        similar=float(similarity(result, count))/x
        valid.append(similar)
        count+=1
    return valid

def cumulatived():
    fitness=[]
    for x in Krom:
        
        valid=validity(x)
        summ=sum(valid)/len(labels_train)
        fitness.append(summ)
    
    cumulative=[]
    prob=[]    
    for x in range(len(Krom)):
        prob.append(fitness[x]/sum(fitness))
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

def fixing_dataset(train, test):
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
    
    train_fix = np.mat(train_fix)
    test_fix = np.mat(test_fix)
    
    return train_fix, test_fix, labels_train, labels_test

if __name__ == '__main__':
#    dataset=raw_input("Masukkan nama data train : ")
    dataset="train_norm.csv"
    test="test_norm.csv"
#    datatest=raw_input("Masukkan nama data test : ");
  #  try :
    train = pd.read_csv(dataset)
    test = pd.read_csv(test)

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
    #splitRatio=0.83333334
    #train, test= splitDataset(train, splitRatio)    
    
    train_fix, test_fix, labels_train, labels_test= fixing_dataset(train, test)

    k=10
    Krom=kromosom()
    #valid=validity(train_fix,labels_train, Krom)
    new_child=[]
    for x in range(1,10):
        print "loop = "+str(x)
        Krom=Krom[:k]+new_child
        prob_fitness, cumulative=cumulatived()
        roulette, Ran=rouletteWheel()        
        Krom=roulette
        cross=crossover(len(labels_train))
        new_child=desimal(cross)
    
    for x in Krom[:k]:
        validity_fix=validity(x)
        print "k        : " + str(x)
        correct= doWork(validity_fix)
        print "Akurasi  : " + str(correct * 100.0) + " % " 
        
#    except IOError as e:
#        print "Tidak ditemukan file"
#    except ValueError:
#        print "Ratio salah"
