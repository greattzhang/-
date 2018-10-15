
"""
Created on Sun Aug 26 19:10:59 2018
#!/usr/bin/python
@author: zhangwn
"""
###
import pandas as pd
import re
from math import isnan    
import numpy as np
###提取内容里面的特征
def process_content(data):
    patt1=re.compile(r'[#【《](.*?)[#】》]',re.S)
    ####为空的值为nan，数据类型为float 可以使用math import isnan 判断
    data['topic']=data['content'].apply(lambda x: patt1.search(x) if type(x)==str else None)
    ###有标题的标注为1，无标题的标注为0 
    data['topic']=data['topic'].apply(lambda x: 0 if x is None else 1)
    ###查看是否有链接
    patt2=re.compile(r'[http]',re.S)
    data['http']=data['content'].apply(lambda x: patt2.search(x) if type(x)==str else None)
    ###有链接的标注为1，无链接的标注为0
    data['http']=data['http'].apply(lambda x: 0 if x is None else 1)
    ###查看是否有@别人
    patt3=re.compile(r'[@]',re.S)
    data['@']=data['content'].apply(lambda x: patt3.search(x) if type(x)==str else None)
    ###有@的标注为1，无@的标注为0
    data['@']=data['@'].apply(lambda x: 0 if x is None else 1) 
    ###查看是否有表情
    patt4=re.compile(r'[[](.*?)[]]',re.S)
    data['emotion']=data['content'].apply(lambda x: patt4.search(x) if type(x)==str else None)
    ###有表情的标注为1，无表情的标注为0
    data['emotion']=data['emotion'].apply(lambda x: 0 if x is None else 1)
    ####含特定词的数目   红包，分享，我们，手气，试试，现金，下载，学习，推荐，代金券，抽到，喜欢，视频，活动，获得，发布，音乐，支持，平台，免费
    da=pd.DataFrame()
    da['wordsum']=[0]*len(data)
    word_list=['红包','分享','我们','手气','试试','现金','下载','学习','推荐','代金券','抽到','喜欢','视频','活动','获得','发布','音乐','支持','平台','免费']    
#    word_list=['数据','技术','学习','开发','设计','用户','应用','代码','安全','系统','互联网','产品','程序员','开源','网络','软件']
#    word_list=['http','@','h']
    for word in word_list:
        patt=[]
        patt=re.compile(word,re.S)
        da[word]=data['content'].apply(lambda x: patt.search(x) if type(x)==str else None)
        da[word]=da[word].apply(lambda x: 0 if x is None else 1) 
        da['wordsum']=da['wordsum']+da[word]
    data['wordsum']=da['wordsum']    
    return data


###求每个用户最大的转发、评论、点赞数
def process_max(data):
    ###求最大值
    df_processmax=data.groupby('uid').agg({'forward_count':np.max,'comment_count':np.max,'like_count':np.max})
#    print (df_processmax.columns)
    df_processmax.columns=['forward_max','comment_max','like_max']
    df_processmax.reset_index(inplace = True)
    data =pd.merge(data, df_processmax, on=['uid']).fillna(0)
    ###求最小值
    df_processmin=data.groupby('uid').agg({'forward_count':np.min,'comment_count':np.min,'like_count':np.min})
    df_processmin.columns=['forward_min','comment_min','like_min']
    df_processmin.reset_index(inplace = True)
    data =pd.merge(data, df_processmin, on=['uid']).fillna(0)
    ####求平均值
    df_processmean=data.groupby('uid').agg({'forward_count':np.mean,'comment_count':np.mean,'like_count':np.mean})
    df_processmean.columns=['forward_mean','comment_mean','like_mean']
    df_processmean.reset_index(inplace = True)
    data =pd.merge(data, df_processmean, on=['uid']).fillna(0)
#####求某一用户发的微博互动大于平均值的概率
    daa=pd.DataFrame({'uid':data['uid'].value_counts()})
    daa.reset_index(inplace=True)
    daa.columns=['uid','count']
    ###统计大于平均值的发博次数
    forward_ave=np.mean(data['forward_count'])
    comment_ave=np.mean(data['comment_count'])
    like_ave=np.mean(data['like_count'])
#    print (forward_ave,comment_ave,like_ave)
    data['forward_judge']=data['forward_count'].apply(lambda x:1 if x>forward_ave else 0)
    data['comment_judge']=data['comment_count'].apply(lambda x:1 if x>comment_ave else 0)
    data['like_judge']=data['like_count'].apply(lambda x:1 if x>like_ave else 0)
    more_ave=data.groupby('uid').agg({'forward_judge':np.sum,'comment_judge':np.sum,'like_judge':np.sum})
    more_ave.columns=['forward_more_ave','comment_more_ave','like_more_ave']
    more_ave.reset_index(inplace = True)
    data.drop(['forward_judge','comment_judge','like_judge'],axis=1, inplace=True)
    daa =pd.merge(more_ave, daa, on=['uid'])
    daa['forward_more_ave_pr']=daa['forward_more_ave']/daa['count']
    daa['comment_more_ave_pr']=daa['comment_more_ave']/daa['count']
    daa['like_more_ave_pr']=daa['like_more_ave']/daa['count']
    daa.drop(['forward_more_ave','comment_more_ave','like_more_ave','count'],axis=1, inplace=True)
    data =pd.merge(data, daa, on=['uid']).fillna(0)  
    data['max_f/l']=pd.DataFrame({'max_f/l':list(map(lambda x, y: x/y if y>0 else 0, data['forward_max'],data['like_max']))})
    data['max_c/l']=pd.DataFrame({'max_c/l':list(map(lambda x, y: x/y if y>0 else 0, data['comment_max'],data['like_max']))})
    data['min_f/l']=pd.DataFrame({'min_f/l':list(map(lambda x, y: x/y if y>0 else 0, data['forward_min'],data['like_min']))})
    data['min_c/l']=pd.DataFrame({'min_c/l':list(map(lambda x, y: x/y if y>0 else 0, data['comment_min'],data['like_min']))})
    data['mean_f/l']=pd.DataFrame({'mean_f/l':list(map(lambda x, y: x/y if y>0 else 0, data['forward_mean'],data['like_mean']))})
    data['mean_c/l']=pd.DataFrame({'mean_c/l':list(map(lambda x, y: x/y if y>0 else 0, data['comment_mean'],data['like_mean']))}) 
    return data


###求发博时间
def process_time(data):
    data['time_date']=data['time'].apply(lambda x: x.date())
###求星期几
    data['time_weekday']=data['time_date'].apply(lambda x: x.weekday())+1
    data['time_weekend1']=((data['time_weekday']==6))
    data['time_weekend2']=((data['time_weekday']==7))
    #data['time_weekend']=(data.loc[1,'time_weekend1'])or(data.loc[1,'time_weekend2'])
    ###计算是否为周末
    data['time_weekend']=pd.DataFrame({'time_weekend':list(map(lambda x, y: 1 if x|y else 0, data['time_weekend1'],data['time_weekend2']))})
###求发博小时
    data['time_hour']=data['time'].apply(lambda x: x.hour)
    data.loc[data.apply(lambda data:(data['time_hour']>-1)and(data['time_hour']<6), axis=1), 'panduan']=1   
    data.loc[data.apply(lambda data:(data['time_hour']>5)and(data['time_hour']<12), axis=1), 'panduan']=2
    data.loc[data.apply(lambda data:(data['time_hour']>11)and(data['time_hour']<18), axis=1), 'panduan']=3 
    data.loc[data.apply(lambda data:(data['time_hour']>17)and(data['time_hour']<24), axis=1), 'panduan']=4
    data.drop(['time_date','time_weekend1','time_weekend2'],axis=1, inplace=True)
    return data

##对预测数据进行处理
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S') 
predict_data=pd.read_table('./Weibo Data/weibo_predict_data.txt',header=None,encoding='utf-8',parse_dates=[2],date_parser=dateparse)
###将预测数据转化为CSV文件
#predict_data.to_csv('./Weibo Data/weibo_predict_data.csv',index=0,encoding='utf_8_sig')
predict_data.columns=['uid','mid','time','content']

predict_data=process_content(predict_data)
predict_data=process_time(predict_data)
predict_data['len_content']=predict_data['content'].apply(lambda x: len(str(x)))
predict_data.to_csv('./Weibo Data/predict_feature.csv',index=0,encoding='utf_8_sig')
#'utf_8_sig'解决CSV写入乱码问题
###查看是否有标题


#对训练数据进行处理
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')       
train_data=pd.read_table('./Weibo Data/weibo_train_data.txt',header=None,encoding='utf-8',parse_dates=[2],date_parser=dateparse)
##将训练数据转化为CSV文件
train_data.to_csv('./Weibo Data/weibo_train_data.csv',index=0,encoding='utf_8_sig')
train_data.columns=['uid','mid','time','forward_count','comment_count','like_count','content']
train_data=process_content(train_data)
#train_data.to_csv('./Weibo Data/train_feature1.csv',index=0,encoding='utf_8_sig')
train_data=process_max(train_data)
#train_data.to_csv('./Weibo Data/train_feature2.csv',index=0,encoding='utf_8_sig')
train_data=process_time(train_data)
train_data['month']=train_data['time'].apply(lambda x: x.month)
train_data['len_content']=train_data['content'].apply(lambda x: len(str(x)))
train_data.to_csv('./Weibo Data/train_feature.csv',index=0,encoding='utf_8_sig')

##对训练数据进行分离
train_data.drop(['forward_max','comment_max','like_max','forward_min','comment_min','like_min','forward_mean','comment_mean','like_mean','forward_more_ave_pr','comment_more_ave_pr','like_more_ave_pr','max_f/l','max_c/l','min_f/l','min_c/l','mean_f/l','mean_c/l'],axis=1, inplace=True)
train_data26=train_data[(train_data['month']==2)|(train_data['month']==3)|(train_data['month']==4)|(train_data['month']==5)|(train_data['month']==6)]
train_data7=train_data[(train_data['month']==7)]
train_data26=process_max(train_data26)
train_data7=process_max(train_data7)
train_data26.to_csv('./Weibo Data/train_feature26.csv',index=0,encoding='utf_8_sig')
train_data7.to_csv('./Weibo Data/train_feature7.csv',index=0,encoding='utf_8_sig')
