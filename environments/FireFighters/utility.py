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

    dim = np.shape(x)[0]
    ce = 0
    for i in range(dim):
        c = 0
        for j in range(np.shape(x)[1]):
            c += np.abs(x[i, j]-y[i, j])**p
        ce += c
    ce = ce / dim
    return ce

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


def one_hot(labels, classes):
    '''
    labels: array of labels
    :param classes:
    :return:
    '''
    output=np.zeros(np.concatenate([np.shape(labels),[classes]]))
    for i in range(np.shape(labels)[0]):
        for j in range(np.shape(labels)[1]):
            output[i,j,int(labels[i,j])]=1
    return output

def cross_entr(x,y):
        dim=np.shape(x)[0]
        ce=0
        for i in range(dim):
            c=0
            for j in range(np.shape(x)[1]):
                c+=-x[i,j]*np.log(y[i,j])
            ce+=c
        ce=ce/dim
        return ce


def index_to_d_set(t, index):
    #Output: the d_set instatiation at time t corresponding to index
    D_set=generate_d_set(t)
    return D_set[index]


def d_set_to_index(t, d_set):
    #Output: the index at time t corresponding to instantiation d_set
    index=np.sum([d_set[t-i]*2**i for i in range(len(d_set))])
    return index

def scomposition(m,n,remainder):
	#return the index m in basis n
    if m < n:
        remainder.insert(0,m)
        return remainder
    else:
        q, r = m//n, m%n
        remainder.insert(0,r)
        scomposition(q,n,remainder)
        return remainder
