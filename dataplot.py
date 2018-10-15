# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 11:59:36 2018

@author: zhangwn
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
train_data=pd.read_csv('./Weibo Data/train_feature.csv',encoding='utf-8') 
##不同点赞、评论、转发数的微博数分布
#转发数为1，2，3.。。的微博个数
data1=train_data.groupby(['forward_count']).size()
data1=data1.reset_index()
data1.columns=['forward_count','count']
print (data1.head())
plt.subplot(3,1,1)
plt.axis([0,200,0,1000])
plt.plot(data1['forward_count'],data1['count'])
plt.title('forward_count')
#print (data1.info())
##点赞数为1，2，3.。。的微博个数
data2=train_data.groupby(['like_count']).size()
data2=data2.reset_index()
data2.columns=['like_count','count']
print (data2.head())
plt.subplot(3,1,2)
plt.axis([0,200,0,1000])
plt.plot(data2['like_count'],data2['count'])
plt.title('like_count')
##评论数为1，2，3.。。的微博个数
data3=train_data.groupby(['comment_count']).size()
data3=data3.reset_index()
data3.columns=['comment_count','count']
print (data3.head())
plt.subplot(3,1,3)
plt.axis([0,200,0,1000])
plt.plot(data3['comment_count'],data3['count'])
plt.title('comment_count')
plt.tight_layout() 
plt.savefig("./plot/不同点赞、评论、转发数的微博频数分布.jpg")
plt.show()

## 用户最大的点赞、评论、转发数的用户频数分布
data=train_data.drop_duplicates(['uid'])
data=data.loc[:,['uid','forward_max','comment_max','like_max']]
data1=data.groupby(['forward_max']).size()
data1=data1.reset_index()
data1.columns=['forward_max','count']
print (data1.head())
plt.subplot(3,1,1)
plt.axis([0,100,0,1000])
plt.plot(data1['forward_max'],data1['count'])
plt.title('forward_max')   

data2=data.groupby(['like_max']).size()
data2=data2.reset_index()
data2.columns=['like_max','count']
plt.subplot(3,1,2)
plt.axis([0,100,0,1000])
plt.plot(data2['like_max'],data2['count'])
plt.title('like_max')   

data3=data.groupby(['comment_max']).size()
data3=data3.reset_index()
data3.columns=['comment_max','count'] 
plt.subplot(3,1,3)
plt.axis([0,100,0,1000])
plt.plot(data3['comment_max'],data3['count'])
plt.title('comment_max')  
plt.tight_layout() 
plt.savefig("./plot/用户最大的点赞、评论、转发数的用户频数分布.jpg")
plt.show()

#发博时间
#周一-周日的发博次数
data=train_data.loc[:,['time_weekday','forward_count','like_count','comment_count']]
data1=data.groupby(['time_weekday']).size()
data1=data1.reset_index()
data1.columns=['time_weekday','count']
plt.figure(figsize=(7, 10), facecolor='#FFFFFF')
plt.subplot(3,1,1)
plt.bar(data1['time_weekday'],data1['count'])
plt.title('weibo_count in weekday') 
##周一-周日的发博的点赞中位数，平均值
data2=data.groupby(['time_weekday']).agg({'forward_count':np.mean,'comment_count':np.mean,'like_count':np.mean})
data2=data2.reset_index()
data2.columns=['time_weekday','forward_mean','comment_mean','like_mean']
print (data2)
plt.subplot(3,1,2)
plt.plot(data2['time_weekday'],data2['forward_mean'],label='forward_mean',color='red')
plt.plot(data2['time_weekday'],data2['like_mean'],label='like_mean',color='green')
plt.plot(data2['time_weekday'],data2['comment_mean'],label='comment_mean',color='blue')
plt.title('behavior_mean in weekday')  
plt.legend()

data3=data.groupby(['time_weekday']).agg({'forward_count':np.max,'comment_count':np.max,'like_count':np.max})
data3=data3.reset_index()
data3.columns=['time_weekday','forward_max','comment_max','like_max']
#print (data3)
plt.subplot(3,1,3)
plt.plot(data3['time_weekday'],data3['forward_max'],label='forward_max',color='red')
plt.plot(data3['time_weekday'],data3['like_max'],label='like_max',color='green')
plt.plot(data3['time_weekday'],data3['comment_max'],label='comment_max',color='blue')
plt.title('behavior_max in weekday')  
plt.legend()
plt.tight_layout() 
plt.savefig("./plot/发博时间对发博次数，最大、平均点赞、转发、评论数的影响.jpg")
plt.show()

##主题、链接、@、表情的影响
data=train_data.loc[:,['topic','http','@','emotion','forward_count','like_count','comment_count']]
#主题
data1=data.groupby(['topic']).agg({'forward_count':np.mean,'comment_count':np.mean,'like_count':np.mean})
data1=data1.reset_index()
data1.columns=['topic','forward_mean','comment_mean','like_mean']
#print (data1[data1['topic']==0])
name_list=['forward_mean','comment_mean','like_mean']
x=list(range(3))
total_width, n = 0.8, 2
width = total_width / n   
data11=list(data1[data1['topic']==0].loc[0,['forward_mean','comment_mean','like_mean']])
print (data11)
plt.figure(figsize=(7, 10), facecolor='#FFFFFF')
plt.subplot(4,1,1)
plt.bar(x,data11,width=width, label='0',fc = 'y')
for i in range(3):
    x[i] = x[i] + width
data11=list(data1[data1['topic']==1].loc[1,['forward_mean','comment_mean','like_mean']])
#print (data11)
plt.bar(x,data11, width=width, label='1',tick_label = name_list,fc = 'r')  
plt.title('topic and behavior')   
#链接
data1=data.groupby(['http']).agg({'forward_count':np.mean,'comment_count':np.mean,'like_count':np.mean})
data1=data1.reset_index()
data1.columns=['http','forward_mean','comment_mean','like_mean']
#print (data1[data1['http']==0])
name_list=['forward_mean','comment_mean','like_mean']
x=list(range(3))
total_width, n = 0.8, 2
width = total_width / n   
data11=list(data1[data1['http']==0].loc[0,['forward_mean','comment_mean','like_mean']])
print (data11)
plt.subplot(4,1,2)
plt.bar(x,data11,width=width, label='0',fc = 'y')
for i in range(3):
    x[i] = x[i] + width
data11=list(data1[data1['http']==1].loc[1,['forward_mean','comment_mean','like_mean']])
#print (data11)
plt.bar(x,data11, width=width, label='1',tick_label = name_list,fc = 'r')  
plt.title('http and behavior')
#@
data1=data.groupby(['@']).agg({'forward_count':np.mean,'comment_count':np.mean,'like_count':np.mean})
data1=data1.reset_index()
data1.columns=['@','forward_mean','comment_mean','like_mean']
#print (data1[data1['http']==0])
name_list=['forward_mean','comment_mean','like_mean']
x=list(range(3))
total_width, n = 0.8, 2
width = total_width / n   
data11=list(data1[data1['@']==0].loc[0,['forward_mean','comment_mean','like_mean']])
print (data11)
plt.subplot(4,1,3)
plt.bar(x,data11,width=width, label='0',fc = 'y')
for i in range(3):
    x[i] = x[i] + width
data11=list(data1[data1['@']==1].loc[1,['forward_mean','comment_mean','like_mean']])
#print (data11)
plt.bar(x,data11, width=width, label='1',tick_label = name_list,fc = 'r')  
plt.title('@ and behavior')

#emotion
data1=data.groupby(['emotion']).agg({'forward_count':np.mean,'comment_count':np.mean,'like_count':np.mean})
data1=data1.reset_index()
data1.columns=['emotion','forward_mean','comment_mean','like_mean']
#print (data1[data1['http']==0])
name_list=['forward_mean','comment_mean','like_mean']
x=list(range(3))
total_width, n = 0.8, 2
width = total_width / n   
data11=list(data1[data1['emotion']==0].loc[0,['forward_mean','comment_mean','like_mean']])
print (data11)
plt.subplot(4,1,4)
plt.bar(x,data11,width=width, label='0',fc = 'y')
for i in range(3):
    x[i] = x[i] + width
data11=list(data1[data1['emotion']==1].loc[1,['forward_mean','comment_mean','like_mean']])
#print (data11)
plt.bar(x,data11, width=width, label='1',tick_label = name_list,fc = 'r')  
plt.title('emotion and behavior')
plt.legend()
plt.tight_layout() 
plt.savefig("./plot/主题、链接、@、表情的影响.jpg")
plt.show()
