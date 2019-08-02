import pandas as pd
import numpy as np
import glob
import os


f = open("48_new.txt")
target = f.read().splitlines()
f.close()

map48 = {}

for i, val in enumerate(target):
    map48[val] = i

print("target: ", target)

def target_array(p):
    listofzeros = [0] * len(target)
    listofzeros[map48[p]] = 1
    return listofzeros




train_x = []
train_y = []



def load_mfcc_from_file(path):
    df=pd.read_csv(path, sep=' ',header=None)
    data = df.values
    size = len(data)

    for x in range(12, size - 12):
        phone = data[x][2]
        if phone in target :
            window = data[x-12:x+13, 3:43]
            train_x.append(window)
            print("insert ")
            train_y.append(target_array(phone))


path2 = '/Users/zhangyousong/Downloads/data/lisa/data/timit/raw/TIMIT/TRAIN2/DR1/FVMH0/SX26.mfcc'



load_mfcc_from_file(path2)
print("shape of train_x" ,np.shape(train_x))
print("shape of train_y" ,np.shape(train_y))



