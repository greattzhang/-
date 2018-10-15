# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:56:44 2018

@author: zhangwn
"""
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
###将之前未出现过的用户，之前转发、评论、点赞最大值为0的用户，出现过且反馈不为0的用户分离开
def class_process(data1,data2):  
    data4=data1.loc[:,['uid','forward_max','comment_max','like_max','forward_min','comment_min','like_min','forward_mean','comment_mean','like_mean','forward_more_ave_pr','comment_more_ave_pr','like_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']]
    data4.drop_duplicates(subset=['uid'],keep='first',inplace=True)
    data4['judge']=pd.DataFrame({'judge':list(map(lambda x,y,z:max(x,y,z),data1['forward_max'],data1['comment_max'],data1['like_max']))})
    data3=data2.loc[:,['uid','mid','topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','panduan']]
    
    data_=pd.merge(data3,data4, on=['uid'],how='inner')
    data_1=data_[data_['judge']!=0]
    
    data=pd.merge(data3,data4, on=['uid'],how='left').fillna(0.01)
    data_2=data[data['judge']==0.01]
#    data_2.drop(['forward_max','comment_max','like_max','forward_min','comment_min','like_min','forward_mean','comment_mean','like_mean'],axis=1,inplace=True)
    data_3=data[data['judge']==0]
    data_1.to_csv('./Weibo Data/data_1.csv',index=0,encoding='utf_8_sig')
    data_2.to_csv('./Weibo Data/data_2.csv',index=0,encoding='utf_8_sig')
    data_3.to_csv('./Weibo Data/data_3.csv',index=0,encoding='utf_8_sig')
    return data_1,data_2,data_3

###对之前未出现过或最大反馈未0的用户都预测0
def predict_23(data1,data2):
    data1['forward_hat']=[0]*len(data1['uid'])
    data1['comment_hat']=[0]*len(data1['uid'])    
    data1['like_hat']=[0]*len(data1['uid'])  
    data2['forward_hat']=[0]*len(data2['uid'])
    data2['comment_hat']=[0]*len(data2['uid'])    
    data2['like_hat']=[0]*len(data2['uid'])  
#    data1.to_csv('./Weibo Data/data_2.csv',index=0,encoding='utf_8_sig')
#    data2.to_csv('./Weibo Data/data_3.csv',index=0,encoding='utf_8_sig')
    return data1,data2

###对出现过反馈的用户进行预测
def predict_1(data1,data2,i):
    clf1 = RandomForestRegressor(criterion='mse', n_estimators =i)
    clf2= RandomForestRegressor(criterion='mse', n_estimators =i)
    clf3 = RandomForestRegressor(criterion='mse',n_estimators =i)

    train_data=data1.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','panduan','forward_max','comment_max','like_max','forward_min','comment_min','like_min','forward_mean','comment_mean','like_mean','forward_more_ave_pr','comment_more_ave_pr','like_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']]
    train_forward=np.array(data1.loc[:,['forward_count']]).ravel()
    train_comment=np.array(data1.loc[:,['comment_count']]).ravel()
    train_like=np.array(data1.loc[:,['like_count']]).ravel()
    predict_data=data2.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','panduan','forward_max','comment_max','like_max','forward_min','comment_min','like_min','forward_mean','comment_mean','like_mean','forward_more_ave_pr','comment_more_ave_pr','like_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']]
    ###进行标准化数据
#    ss = StandardScaler()
#    train_data = pd.DataFrame(ss.fit_transform(train_data),columns=['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','panduan','forward_max','comment_max','like_max','forward_min','comment_min','like_min','forward_mean','comment_mean','like_mean','forward_more_ave_pr','comment_more_ave_pr','like_more_ave_pr'])
#    
#    predict_data =pd.DataFrame( ss.transform(predict_data),columns=['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','panduan','forward_max','comment_max','like_max','forward_min','comment_min','like_min','forward_mean','comment_mean','like_mean','forward_more_ave_pr','comment_more_ave_pr','like_more_ave_pr'])
    predict_forward=clf1.fit(train_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','forward_max','forward_min','forward_mean','forward_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']],train_forward)
    forward_hat=np.round(predict_forward.predict(predict_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','forward_max','forward_min','forward_mean','forward_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']]),0)
    predict_comment=clf2.fit(train_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','comment_max','comment_min','comment_mean','comment_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']],train_comment)
    comment_hat=np.round(predict_comment.predict(predict_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','comment_max','comment_min','comment_mean','comment_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']]),0)
    predict_like=clf3.fit(train_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','like_max','like_min','like_mean','like_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']],train_like)
    like_hat=np.round(predict_like.predict(predict_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','like_max','like_min','like_mean','like_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']]),0)

  

#    predict_forward=clf.fit(train_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','forward_max','forward_min','forward_mean','forward_more_ave_pr','max_f/l','min_f/l','mean_f/l']],train_forward)
#    forward_hat=np.round(predict_forward.predict(predict_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','forward_max','forward_min','forward_mean','forward_more_ave_pr','max_f/l','min_f/l','mean_f/l']]),0)
#    predict_comment=clf.fit(train_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','comment_max','comment_min','comment_mean','comment_more_ave_pr','max_c/l','min_c/l','mean_c/l']],train_comment)
#    comment_hat=np.round(predict_comment.predict(predict_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','comment_max','comment_min','comment_mean','comment_more_ave_pr','max_c/l','min_c/l','mean_c/l']]),0)
#    predict_like=clf.fit(train_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','like_max','like_min','like_mean','like_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']],train_like)
#    like_hat=np.round(predict_like.predict(predict_data.loc[:,['topic','http','@','emotion','wordsum','len_content','time_weekend','time_weekday','time_hour','like_max','like_min','like_mean','like_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l']]),0)
    data2['forward_hat']=forward_hat
    data2['comment_hat']=comment_hat 
    data2['like_hat']=like_hat
    data2['forward_hat']=data2['forward_hat'].apply(lambda x:int(x))
    data2['comment_hat']=data2['comment_hat'].apply(lambda x:int(x))
    data2['like_hat']=data2['like_hat'].apply(lambda x:int(x))    
    return data2

###计算准确率
def precision(data):
    data['deviation_forward']=list(map(lambda x, y: abs(x-y)/(y+5), data['forward_hat'],data['forward_count']))
#print (data['deviation_forward'])
    data['deviation_like']=list(map(lambda x, y: abs(x-y)/(y+3), data['like_hat'],data['like_count']))
    #print (data['deviation_like'])
    data['deviation_comment']=list(map(lambda x, y: abs(x-y)/(y+3), data['comment_hat'],data['comment_count']))
    #print (data['deviation_comment'])
    data['lcf_sum']=data['forward_count']+data['like_count']+data['comment_count']
#    print (data['lcf_sum'])
    data['lcf_sum']=data['lcf_sum'].apply(lambda x: 100 if x>100 else x)
    data['precision_1_-0.8']=1-0.5*data['deviation_forward']-0.25*data['deviation_like']-0.25*data['deviation_comment']-0.8
    #print (data['precision_1_-0.8'])
    data.loc[data['precision_1_-0.8']<=0,'sgn']=0
    data.loc[data['precision_1_-0.8']>0,'sgn']=1
#    print (data['sgn'])
    precision_=sum((data['lcf_sum']+1)*data['sgn'])/sum(data['lcf_sum']+1)

    
    return precision_

train_data26=pd.read_csv('./Weibo Data/train_feature26.csv',encoding='utf-8')    
train_data7=pd.read_csv('./Weibo Data/train_feature7.csv',encoding='utf-8')
train_data7_=train_data7.drop(['forward_max','comment_max','like_max','forward_min','comment_min','like_min','forward_mean','comment_mean','like_mean','forward_more_ave_pr','comment_more_ave_pr','like_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l'],axis=1,inplace=False)

data_1,data_2,data_3=class_process(train_data26,train_data7_)
data_2,data_3=predict_23(data_2,data_3)
for i in np.arange(10,110,9):
    data_1_=predict_1(train_data26,data_1,i)
    result=pd.concat([data_1_.loc[:,['uid','mid','forward_hat','comment_hat','like_hat']],data_2.loc[:,['uid','mid','forward_hat','comment_hat','like_hat']],data_3.loc[:,['uid','mid','forward_hat','comment_hat','like_hat']]])
#    result.to_csv('./Weibo Data/test_result.csv',index=0,encoding='utf_8_sig')
    result=pd.merge(train_data7,result.loc[:,['uid','mid','forward_hat','comment_hat','like_hat']],on=['uid','mid'],how='inner')
#    train_data7.to_csv('./Weibo Data/test_result.csv',index=0,encoding='utf_8_sig')
    print ('depth',i,'leaf',3,precision(result))

