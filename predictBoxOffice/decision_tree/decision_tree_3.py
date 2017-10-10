# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:25:22 2017

@author: Administrator
"""

import numpy as np
import pandas
import random
import pickle


class ccDecisionTree():
#==============================================================================
#     data表示数据其中前几列表示x，最后1列表示y
#     labels表示每一列对应的属性名称
#     isDiscrete表示每一列是否是离散属性，其中最后一列应该用不到
#     min_samples_leaf整数表示如果叶子节点中的数据个数小于等于其数值则停止生长
#     max_samples_leaf浮点数表示如果叶子节点中有某一结果占有比例大于其值则停止生长
#     criterion表示求混乱度的标准、方法
#==============================================================================
    def __init__(self,data,labels=None,isDiscrete=None,min_samples_leaf=0,max_samples_leaf=1,max_depth=100,criterion='entropy'):
#        在数据第一列加一列序号列,同时labels最前面加上序号列的label，分别存为私有变量使用__data,__labels
        if type(data)==list:
            self.data=np.array(data)
        else:
            self.data=data.copy()
        indexColumn=np.array([x for x in range(self.data.shape[0])]).reshape((-1,1))
        self.__data=np.hstack((indexColumn,self.data))

        
        if labels:
            self.labels=labels.copy()
        else:
            self.labels=['Attri'+str(i) for i in range(self.data.shape[-1])]
        self.__labels=['Index']
        self.__labels.extend(self.labels)
        
        if isDiscrete:
            self.__isDiscrete=isDiscrete.copy()
        else:
            self.__isDiscrete=[self.__judgeDiscrete(self.data,i) for i in range(self.data.shape[-1])]    #1表示离散 0表示连续
        self.__isDiscrete.insert(0,1)
        self.max_depth=max_depth
        self.min_samples_leaf=min_samples_leaf
        self.max_samples_leaf=max_samples_leaf
        self.criterion=criterion
        self.nodeInfoDict={}
        self.nowNodeNums=0
        self.dont_need_tree_flag=0
        self.tree=self.__createTree(self.__data,self.__labels)
        

    def __judgeDiscrete(self,data,column):
        return np.unique(data[:,column]).shape[0]<5

    def __majorityY(self,classList):
        return np.argmax(np.bincount(classList))
        
    def __splitDataSet(self,dataSet, bestFeatIndex, value,symbol,keep_original_column):
        if symbol=='==':
            local_data=dataSet[dataSet[:,bestFeatIndex]==value]
        elif symbol=='<=':
            local_data=dataSet[dataSet[:,bestFeatIndex]<=value]  #返回分类后的新矩阵
        else:
            local_data=dataSet[dataSet[:,bestFeatIndex]> value]
            if symbol!='>':
                print('splitDataSet symbol error',symbol)
        if keep_original_column:
            return local_data
        else:
            return np.delete(local_data,bestFeatIndex,axis=1)
    
    def __chooseBestFeatureToSplit(self,dataSet,labels_local):
        baseEntropy = self.__calcShannonEnt(dataSet)
        bestInfoGain = -1; bestFeature = -1
        splitPointContinue=-1
        for i in range(1,dataSet.shape[1]-1):  #求所有属性的信息增益
#            featList = [example[i] for example in dataSet]
            isDiscreteFlag=self.__isDiscrete[self.__labels.index(labels_local[i])]
#           离散isDiscreteFlag==1
            if isDiscreteFlag:
                uniqueVals = set(dataSet[:,i])
                newEntropy = 0.0  
                splitInfo = 0.0;
                for value in uniqueVals:  #求第i列属性每个不同值的熵*他们的概率
                    subDataSet = self.__splitDataSet(dataSet, i , value,'==',0)  
                    prob = subDataSet.shape[0]/dataSet.shape[0]  #求出该值在i列属性中的概率
                    newEntropy += prob * self.__calcShannonEnt(subDataSet)  #求i列属性各值对于的熵求和
                    splitInfo -= prob * np.log2(prob)
#               注意夜里prob肯定不为0所以不会报错，但是当uniqueVals只有一个元素是splitInfo和baseEntropy - newEntropy都是0，相除会报错，
#               因此用函数解决同时方便后面改标准
                infoGain=self.__calInfoGainRatio(baseEntropy,newEntropy,splitInfo,isDiscreteFlag)
#                infoGain = (baseEntropy - newEntropy) / splitInfo  #求出第i列属性的信息增益率
                splitPointContinueNow=-1
    #            print('infoGain',infoGain,len(uniqueVals));
            else:
                splitPointContinueNow,newEntropy,splitInfo=self.__getBestSplitConuetine(dataSet,i)
#                dataSet[:,i]把第i列的数据传进去算log2(N-1)/D
                infoGain=self.__calInfoGainRatio(baseEntropy,newEntropy,splitInfo,isDiscreteFlag,dataSet[:,i])
            if(infoGain > bestInfoGain):  #保存信息增益率最大的信息增益率值以及所在的下表（列值i）
                bestInfoGain = infoGain  
                bestFeature = i
                splitPointContinue=splitPointContinueNow
        return bestFeature,splitPointContinue
    

    def __getBestSplitConuetine(self,dataSet,i):
        splitInfo = 0.0;
        minEntropy=999
        value=-1
        if dataSet.shape[0]==1:
            print('getBestSplitConuetine error')
        uniqueSplitPoint=(np.unique(dataSet[:,i])[:-1]+np.unique(dataSet[:,i])[1:])/2
        for splitPoint in uniqueSplitPoint:
            subDataSet1 = self.__splitDataSet(dataSet, i , splitPoint,'<=',1)
            subDataSet2 = self.__splitDataSet(dataSet, i , splitPoint,'>',1)
            p1=subDataSet1.shape[0]/(subDataSet1.shape[0]+subDataSet2.shape[0])
            p2=1-p1
            newEntropy=p1*self.__calcShannonEnt(subDataSet1)+p2*self.__calcShannonEnt(subDataSet2)
            if newEntropy<minEntropy:
                minEntropy=newEntropy
                value=splitPoint
                splitInfo=-p1*np.log2(p1)-p2*np.log2(p2)
        return value,minEntropy,splitInfo
    
    def __calcShannonEnt(self,dataSet):
        count_label=np.bincount(dataSet[:,-1])  
        count_label_nonzero=count_label[count_label.nonzero()]
        count_label_p=count_label_nonzero/np.sum(count_label_nonzero)
        if self.criterion=='entropy':
            shannonEnt=np.sum(-count_label_p*np.log2(count_label_p))
        elif self.criterion=='gini':
            shannonEnt=np.sum(1-count_label_p**2)
        return shannonEnt;
    
    def __calInfoGainRatio(self,baseEntropy,newEntropy,splitInfo,isDiscreteFlag,dataSetI=None):
        if self.criterion=='entropy':
            InfoGain=baseEntropy-newEntropy
            if splitInfo==0:
                return 0
    #        离散
            elif isDiscreteFlag:
                return InfoGain/splitInfo
    #        连续变量
            else:
#                return InfoGain
#                return InfoGain/splitInfo
                return InfoGain-(np.log2(np.unique(dataSetI).shape[0]-1)/dataSetI.shape[0])*(dataSetI.shape[0]!=0)
        else:
            return baseEntropy-newEntropy
        
    
    def __createTree(self,data=None,labels=None,now_depth=1):
#        nowIndex=data[:,0].tolist()
#        nowLabelFromIndex=self.__getLabel(nowIndex)
#        print(now_depth)
        labels_local=labels[:]
        classList = data[:,-1]
        self.nodeInfoDict['Node'+str(self.nowNodeNums)]={'index':data[:,0].tolist(),'label':data[:,-1].tolist(),'most_label':self.__majorityY(classList)}
        self.nowNodeNums+=1
#        只剩一种结果
        if len(set(classList))==1:
            return 'Node'+str(self.nowNodeNums-1)+'_'+str(self.__majorityY(classList));
#        没有属性可分，只剩序号列和结果列
        if data.shape[1]==2:
            return 'Node'+str(self.nowNodeNums-1)+'_'+str(self.__majorityY(classList));
        if data.shape[0]<self.min_samples_leaf:
            return 'Node'+str(self.nowNodeNums-1)+'_'+str(self.__majorityY(classList));
        if np.max(np.bincount(classList))/len(classList)>self.max_samples_leaf:
            if now_depth>1:
                return 'Node'+str(self.nowNodeNums-1)+'_'+str(self.__majorityY(classList));
            else:
                self.dont_need_tree_flag=1
                print('max_samples_leaf 太小了')
                return np.argmax(np.bincount(classList))
                
        if now_depth>=self.max_depth:
            return 'Node'+str(self.nowNodeNums-1)+'_'+str(self.__majorityY(classList));
#       返回值是1~n-2，第0列是序号列，最后一列是结果列
        bestFeatIndexLocal,splitPointForContinue = self.__chooseBestFeatureToSplit(data,labels_local)
        bestFeatLabel = labels_local[bestFeatIndexLocal]
        bestFeatIndexGlobal=self.__labels.index(bestFeatLabel)
#        print(bestFeatIndex,bestFeatLabel)
        nodeName='Node'+str(self.nowNodeNums-1)+'_'+bestFeatLabel
#        self.nowNodeNums+=1
        myTree = {nodeName:{}}
#       离散变量时
        if self.__isDiscrete[bestFeatIndexGlobal]==1:
            del(labels_local[bestFeatIndexLocal])  #从属性列表中删掉已经被选出来当根节点的属性
            uniqueVals = set(data[:,bestFeatIndexLocal])  #求出该属性的所有值得集合（集合的元素不能重复）
            for value in uniqueVals:  #根据该属性的值求树的各个分支
                subLabels = labels_local[:]  
                myTree[nodeName]["== "+str(value)] = self.__createTree(self.__splitDataSet(data, bestFeatIndexLocal, value,'==',0), subLabels,now_depth+1)  #根据各个分支递归创建树
            return myTree
        else:
            for symbol in ['<=','>']:
                subLabels = labels_local[:]
                dataSubSet=self.__splitDataSet(data, bestFeatIndexLocal, splitPointForContinue,symbol,1)
                if dataSubSet.shape[0]!=0:
                    myTree[nodeName][symbol+' '+str(splitPointForContinue)] = self.__createTree(dataSubSet, subLabels,now_depth+1)
            return myTree
        
    def classify(self,data):
        if self.dont_need_tree_flag==1:
            return self.tree
        tree=self.tree
        l=[]
        for dataone in data:
            l.append(int(self.__classifyone(tree,dataone).split('_')[1]))
        return l
        
    def __classifyone(self,tree,dataone):
#        print('--------')
        if type(tree).__name__ == 'str':
            print(tree)
        nodeNameAll = list(tree.keys())[0]
#        print(nodeName)
        nodeName,label=nodeNameAll.split('_')
        nextDict = tree[nodeNameAll]
        featIndex=self.__labels.index(label)
#       减一是因为内部多了一列序号列而测试数据没有这一列
        isDiscreteFlag=self.__isDiscrete[featIndex]
        keyList=list(nextDict.keys())
        featIndex-=1
#       离散属性情况
        if isDiscreteFlag:
            flag=0
            for key in keyList:
#                print(featIndex)
#                print(dataone[featIndex])
#                print(key)
#                print(int(key.split('_')[1]))
                if dataone[featIndex]==int(key.split(' ')[1]):
                    flag=1
                    if type(nextDict[key]).__name__ == 'dict':
                        classLabel = self.__classifyone(nextDict[key],dataone)
                    else:
                        classLabel = nextDict[key]
#           出现找不到的情况去父节点里找最多的类归为自己的类别
            if flag==0:
                num_need=self.nodeInfoDict[nodeName]['most_label']
                classLabel='**_'+str(num_need)
        else:
            item,numStr=keyList[0].split(' ')
            if dataone[featIndex]>int(float(numStr)):
                key=keyList[0] if '>' in keyList[0] else keyList[1]
            else:
                key=keyList[0] if '<=' in keyList[0] else keyList[1]
            if type(nextDict[key]).__name__ == 'dict':  
                classLabel = self.__classifyone(nextDict[key],dataone)
            else:
                classLabel = nextDict[key]
        return classLabel
        
def read():
    r=pandas.read_csv(r'F:\mine\code\movie\2017.7.10dtree\spss_2.csv',encoding='utf-8')
    name=r.keys().tolist()
    data=r.values.tolist()
    return data,name

def mysplit(data,test_size):
    train=[]
    test=[]
    n=len(data)
    l_test_index=random.sample(range(n),int(n*test_size))
    for index,i in enumerate(data):
        if index in l_test_index:
            test.append(i)
        else:
            train.append(i)
    return train,test

def make_n_labels_same_space(y,n):
    yy=y.copy()
    yy.sort()
    min_element=yy[0]
    max_element=yy[-2]
    per_length=(max_element-min_element)/n
    l=[round(min_element+per_length*i) for i in range(1,n)]
#    print(l)
    for i,j in enumerate(y):
        t=0
        for k in l:
            if y[i]>k:
                t+=1
        yy[i]=t
    return yy

def together_xy(x,y):
    yy=y.copy()
    return np.hstack((x,yy.reshape((-1,1))))

def average_split_train_test(xy,p):
    labels=xy[:,-1]
    nums=np.bincount(labels)
    test_nums=np.round(nums*p)
    l_test=[]
    for i in np.unique(labels):
        l_test.extend(random.sample(np.where(labels==i)[0].tolist(),int(test_nums[i])))
    train=[]
    test=[]
    for index,i in enumerate(xy):
        if index in l_test:
            test.append(i)
        else:
            train.append(i)
    return train,test


class data:
    def __init__(self,x,y,name):
        self.x=x
        self.y=y
        self.name=name


if __name__=='__main__':
    
    with open(r'.\data.pkl','rb') as f:
        data_all=pickle.load(f)
    
    y=data_all.y
    x=data_all.x
    y=np.delete(y,np.where(x[:,-1]<0)[0],0)
    x=np.delete(x,np.where(x[:,-1]<0)[0],0)
    y_labels=make_n_labels_same_space(y,6)
    xy=together_xy(x,y_labels)
    l=[]
    for _ in range(1):
        train,test=average_split_train_test(xy,0.3)
#        l_isDisctete=[1,1,0,0,1,1,0,0,0,1,0,0,0,0]
#        tree=ccDecisionTree(train,data_all.name,isDiscrete=l_isDisctete)
        tree=ccDecisionTree(train,data_all.name)
        pre_y=np.array(tree.classify(test))
        test_y=np.array(test)[:,-1]
        l.append(sum(pre_y==test_y)/len(pre_y))
#        print(sum(pre_y==test_y)/len(pre_y))
    print('last result:')
    print(np.mean(np.array(l)))
#    with open(r'tree_for_python2_0.pkl','wb') as f:
#        pickle.dump(tree.tree,f,2)
#    with open(r'node_info_for_python2_0.pkl','wb') as f:
#        pickle.dump(tree.nodeInfoDict,f,2)
#    
#==============================================================================
#     n=100
#     
#     min_x=5
#     max_x=0.7
#     max_depth=10
#     #for min_x in range(5,6):
#     for max_depth in range(3,7):
#         for max_x in [0.6,0.7,0.8,0.9,1]:
#             l=[]
#             for i in range(n):
#                 train,test=mysplit(data,0.3)
#             #    tree=ccDecisionTree(train,name)
#                 tree=ccDecisionTree(train,name,min_samples_leaf=min_x,max_samples_leaf=max_x,max_depth=max_depth,criterion='entropy')
#                 pre_y=np.array(tree.classify(test))
#                 test_y=np.array(test)[:,-1]
#                 #print(sum(pre_y==test_y)/len(pre_y))
#                 l.append(sum(pre_y==test_y)/len(pre_y))
#             print('p: {:.2f} min_x: {} max_x: {} max_depth: {}'.format(sum(l)/n,min_x,max_x,max_depth))
#==============================================================================

