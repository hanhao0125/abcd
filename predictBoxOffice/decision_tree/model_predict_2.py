# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 22:46:24 2017

@author: Administrator
"""
import pickle
import numpy as np

def get_average_core(l_actor,actor_dict):
    l_core=[]
    for one in l_actor:
        if one in actor_dict:
            l_core.append(actor_dict[one])
    if len(l_core)>0:
        return int(sum(l_core)/len(l_core))
    else:
        return 0

def stringToNum(filmName,cost,theme,director,actor,releaseDate):
    with open(r'director.pkl','rb') as f:
        director_dict=pickle.load(f)
    with open(r'actor.pkl','rb') as f:
        actor_dict=pickle.load(f)
    with open(r'genre.pkl','rb') as f:
        genre_dict=pickle.load(f)
    l=[]
    l.append(int(releaseDate.split('.')[1]))
    if director in director_dict:
        l.append(round(director_dict[director]))
    else:
        l.append(0)
    l.append(get_average_core(actor.split(','),actor_dict))
    l.append(get_average_core(theme.split(','),genre_dict))
    l.append(int(cost))
    return l

def predict(l):
    with open('tree2.pkl','rb') as f:
        tree=pickle.load(f)
    l_all=[l]
    return tree.classify(np.array(l_all))[0]

if __name__=='__main__':
    l=stringToNum(filmName="速度与激情8",cost='175',theme="动作,犯罪",director="F·加里·格雷", actor="范·迪塞尔,道恩·强森,查理兹·塞隆,杰森·斯坦森,米歇尔·罗德里格兹", \
                releaseDate='2017.4.14')
    label=predict(l)
    print(label)
    
    l_for_labels=[0,0.5,1,1.5,2,3,4,5,10,20,1000]
    if label<9:
        print("{}亿~{}亿".format(l_for_labels[label],l_for_labels[label+1]))
    else:
        print("{}+亿".format(l_for_labels[label]))
    
    