#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*- s


import os
import _pickle
import numpy as np

def index_to_inst(n,base):
    return lambda x: np.concatenate(([0 for i in range(n-len(np.base_repr(x,base=base)))],[int(i) for i in np.base_repr(x,base=base)])).astype('int')

def inst_to_index(base):
    return lambda x: np.sum([x[-i - 1] * (base ** i) for i in range(len(x))])


def KL_divergence(p,q):
    assert len(p)==len(q)
    p=np.array(p).astype(float)
    q=np.array(q).astype(float)
    KL=0
    for i in range(len(p)):
        if p[i]==0:
            KL+=0
        elif q[i]==0:
            print ('kl_divergence is infty')
            return
        else:
            KL+=p[i]*np.log(p[i]/q[i])

    return KL

def norm_p(x,y,p=1):
    
    dim=np.shape(x)[0]
    n_outputs=np.shape(x)[1]
    n_classes=np.shape(x)[2]
    ce=np.zeros(n_outputs)
    for i in range(dim):
        for o in range(n_outputs):
            for j in range(n_classes):
                ce[o]+=np.abs(x[i, o, j]-y[i,o, j])**p      
    ce=ce/dim  
    return np.sum(ce)

def TotVar(p,q):

    if not np.any(p) or not np.any(q):
        return 0
    else:
        return (np.abs(np.array(p)-np.array(q))).sum()



def save_obj(obj, filename="obj.bin", protocol=3):
    """
    Dumps obj Python to a file using cPickle.

    :Parameters:
        obj : object Python
        filename : str
            Path to the file where obj is dumped
    """
    if  not  os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    file = open(filename, 'wb')
    _pickle.dump(obj, file, protocol)
    file.close()
    return




def load_obj(filename="obj.bin"):
    """
    Loads obj Python pickled previously with `save_obj`.

    :Parameters:
        filename : str
            Path to the file with saved save_obj
    """
    file = open(filename, 'rb')
    obj = _pickle.load(file)
    return obj


def one_hot(labels,classes):
    '''
    labels: array of labels
    :param classes:
    :return:
    '''
    output=np.zeros(np.concatenate([np.shape(labels),[classes]]))
    for i in range(np.shape(labels)[0]):
        for j in range(np.shape(labels)[1]):
            for k in range(np.shape(labels)[2]):
                output[i,j,k,int(labels[i,j,k])]=1
    return output

def cross_entr(x,y):
        dim=np.shape(x)[0]
        n_outputs=np.shape(x)[1]
        n_classes=np.shape(x)[2]
        ce=np.zeros(n_outputs)
        for i in range(dim):
            for o in range(n_outputs):
                for j in range(n_classes):
                    ce[o]+=-x[i,o,j]*np.log(y[i,o,j])        
        ce=ce/dim   
        return np.sum(ce)


def index_to_d_set(t, index):
    #Output: the d_set instatiation at time t corresponding to index
    D_set=generate_d_set(t)
    return D_set[index]


def d_set_to_index(t, d_set):
    #Output: the index at time t corresponding to instantiation d_set
    index=np.sum([d_set[t-i]*2**i for i in range(len(d_set))])
    return index


def generate_d_set(t):
    #Generate all the possible instantiation of the D_set at time step t
    D_set=[]
    for i in range(2**(t+1)):
        d = "{0:b}".format(i)
        d = np.array([int(j) for j in d])
        while(len(d)<=t):
            d=np.insert(d,0,0,axis=0)
        D_set.append(d)

    return np.array(D_set)


