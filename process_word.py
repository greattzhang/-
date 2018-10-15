# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:56:44 2018

@author: zhangwn
"""
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import jieba
train_data=pd.read_csv('./Weibo Data/weibo_train_data.csv',encoding='utf-8')
train_data.columns=['uid','mid','time','forward_count','comment_count','like_count','content']
train_data['sum']=train_data['forward_count']+train_data['comment_count']+train_data['like_count']
train_data_=train_data.sort_index(axis = 0,ascending = False,by = 'sum')
#print (train_data_)
print (len(train_data['sum']))
train_data_.loc[:60000,'content'].to_csv('./Weibo Data/word.csv',index=0,header=0,encoding='utf_8_sig')

test = open('./Weibo Data/word.csv',encoding='utf-8')

word = open('./Weibo Data/word1.txt', 'w',encoding='utf-8')
for line in test:
#    print (line)
    line = line.strip()
    try:
        content=line
#        uid,mid,time,a,b,c,content=line.split(",",6)
    except:
        content=''
    word.write(str(content))
test.close()    
word.close()
test4=open('./Weibo Data/word1.txt', encoding='utf-8')
test3 =test4.read()
print (len(test3))
santi_words = [x for x in jieba.cut(test3) if len(x) >= 2]
c = Counter(santi_words).most_common(100)
print (c)
test4.close() 
