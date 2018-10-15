# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 20:10:22 2018

@author: zhangwn
"""
from datetime import *
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import   dtpredict  as pt
starttime = datetime.now()
train_data=pd.read_csv('./Weibo Data/train_feature.csv',encoding='utf-8')    
predict_data=pd.read_csv('./Weibo Data/predict_feature.csv',encoding='utf-8')
data_1,data_2,data_3=pt.class_process(train_data,predict_data)
data_2,data_3=pt.predict_23(data_2,data_3)

data_1_=pt.predict_1(train_data,data_1,10,3)
result=pt.pd.concat([data_1_.loc[:,['uid','mid','forward_hat','comment_hat','like_hat']],data_2.loc[:,['uid','mid','forward_hat','comment_hat','like_hat']],data_3.loc[:,['uid','mid','forward_hat','comment_hat','like_hat']]])
#    result.to_csv('./Weibo Data/test_result.csv',index=0,encoding='utf_8_sig')
result=pd.merge(predict_data,result.loc[:,['uid','mid','forward_hat','comment_hat','like_hat']],on=['uid','mid'],how='inner')
#    train_data7.to_csv('./Weibo Data/test_result.csv',index=0,encoding='utf_8_sig')
result=result.loc[:,['uid','mid','forward_hat','comment_hat','like_hat']] 
result.to_csv('./Weibo Data/result0904.csv',index=0,header=0,encoding='utf_8_sig')
result = open('./Weibo Data/result0904.csv',encoding='utf-8')
weibo_result = open('./Weibo Data/weibo_result0904.txt', 'w',encoding= 'utf-8')
for line in result:
    line = line.strip()
    uid,mid,forward,comment,like=line.split(",")
    weibo_result.write(uid+"\t"+mid+"\t"+forward+ ","+comment+ ","+like+ "\n")
weibo_result.close()    
result.close()
#final=pd.DataFrame()
#for index in range(len(result['uid'])):
#    final.loc[index,'a']=str(result.loc[index,'uid'])+'\t'+str(result.loc[index,'mid'])+'\t'+str(result.loc[index,'forward_count'])+','+str(result.loc[index,'comment_count'])+','+str(result.loc[index,'like_count'])
#    weibo_result .write(final.loc[index,'a']+'\n')
#weibo_result.close() 
endtime = datetime.now()
print ("Total running time: %f s" % (endtime - starttime).seconds)                              