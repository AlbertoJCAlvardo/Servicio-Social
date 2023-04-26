#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:18:25 2023

@author: alberto
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack


from sklearn.neural_network import MLPClassifier     #Libreria Perceptron Multicapa

def preprocesar(x_train,x_test,bina):
    vectorizer = CountVectorizer(binary=bina, ngram_range = (1,3))
    x_tr = vectorizer.fit_transform(x_train)
    x_te = vectorizer.transform(x_test)
    return x_tr, x_te

def agregar_features(x_tr,x_te,features,features_test):
    x_train = hstack((x_tr, features))
    x_test = hstack((x_te, features_test))
    return x_train,x_test

def metricas(y_te,y_pred):
    p = precision_score(y_te, y_pred, average = "macro")
    r = recall_score(y_te, y_pred, average = "macro")
    a = accuracy_score(y_te, y_pred)
    f = f1_score(y_te,y_pred, average = "macro")
    return p,r,a,f

def imprimir(l,f):
    aux = np.array(l)
    f.write(str(round(aux.mean(),4)))
    f.write("\n")

df = pd.read_csv("train_c.csv")
lemma = list(df["lemma"])
xpos = list(df["xpos"])
y = np.array(list(df["num_label"]))

kf = KFold(n_splits=10, shuffle=True)

acc_rl = list()
rec_rl = list()
pre_rl = list()
f1_rl = list()

acc_svm = list()
rec_svm = list()
pre_svm = list()
f1_svm = list()

for train, test in kf.split(lemma):
    #x_train = list()
    #x_test = list()
    lemma_train = list()
    xpos_train = list()
    lemma_test = list()
    xpos_test = list()

    for i in train:
        #x_train.append(X[i])
        lemma_train.append(lemma[i])
        xpos_train.append(xpos[i])
    for i in test:
        #x_test.append(X[i])
        lemma_test.append(lemma[i])
        xpos_test.append(xpos[i])
    y_train, y_test = y[train], y[test]
    
    #x_tr, x_te = preprocesar(x_train,x_test)
    #x_tr, x_te = preprocesar(x_train,x_test)
    lemma_tr, lemma_te = preprocesar(xpos_train,xpos_test,True)
    xpos_tr, xpos_te = preprocesar(lemma_train,lemma_test,False)
    
    x_tr, x_te = agregar_features(lemma_tr, lemma_te, xpos_tr, xpos_te)
    
    mlp = MLPClassifier(hidden_layer_sizes=(50,),activation="logistic",solver="lbfgs")

    mlp.fit(x_tr,y_train)
    
    pred = mlp.predict(x_te)

    p,r,a,f = metricas(pred,y_test)
    pre_rl.append(p)
    rec_rl.append(r)
    acc_rl.append(a)
    f1_rl.append(f)



    f = open('trigramas_pos+uni_mlp.txt',"w")
    f.write("MLP (100 hidden layers) (logistic activation function) \n")
    f.write("recall")
    imprimir(rec_rl,f)
    f.write("precision")
    imprimir(pre_rl,f)
    f.write("acc")
    imprimir(acc_rl,f)
    f.write("f1")
    imprimir(f1_rl,f)

    


    f.close()
