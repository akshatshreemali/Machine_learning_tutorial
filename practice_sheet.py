
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('C:/Users/Akshat/Downloads/Noise_Data_new.csv')


# In[8]:


import glob
path ='C:/Users/Akshat/Desktop/Interview'
filenames = glob.glob( path+"/*.csv")


# In[9]:


filenames


# In[12]:


big_frame = pd.concat([pd.read_csv(f, sep=',') for f in glob.glob(path + "/*.csv")],
                      ignore_index=True)


# In[13]:


big_frame.head()


# In[14]:


# s=pd.concat([pd.read_csv(f,sep=',') for f in glob.glob(path+"/*.csv")],ignore_index=True)


# In[15]:


data.head()


# In[16]:


data.dtypes


# In[19]:


data.shape


# In[3]:


data.isnull().sum()


# In[4]:


data=data.dropna(axis=1,how='all')


# In[5]:


data.shape[1]


# In[6]:


data.drop(['Landmark'],axis=1,inplace=True)


# In[7]:


data.drop(['Latitude','Longitude'],axis=1,inplace=True)


# In[32]:


data.shape[1]


# In[33]:


data[data.duplicated(keep=False)]


# In[35]:


data=data.drop_duplicates()


# In[36]:


data.shape


# In[37]:


data.head()


# In[38]:


data['School Number'].value_counts()


# In[39]:


data['Agency'].value_counts()


# In[40]:


data['School Number'].unique()


# In[41]:


data['School Not Found']=np.where(data['School Not Found']=='N',0,1)


# In[42]:


data['School Not Found'].unique()


# In[47]:


data['Created Date'] = pd.to_datetime(data['Created Date'])
data['Closed Date'] = pd.to_datetime(data['Closed Date'])
data['Resolution']=(data['Closed Date']- data['Created Date']).dt.days


# In[48]:


data_1=data.sample(frac=0.3)


# In[43]:


import seaborn as sns


# In[49]:


data_1['Resolution'].unique()


# In[51]:


sns.boxplot(x=data_1['School Not Found'],y=data_1['Resolution'],data=data)


# In[117]:


fig,ax=plt.subplots(figsize=(12,6))
sns.barplot(x=data_1['School Not Found'],y=data_1['Resolution'],data=data)


# In[3]:


data=pd.read_csv('C:/Users/Akshat/Downloads/ozan_p_pApply_intern_challenge_03_26_min.csv')


# In[5]:


data=data.sample(frac=0.2)


# In[6]:


data.shape


# In[57]:


data.columns


# In[58]:


data.isnull().sum()


# In[59]:


data.head()


# In[63]:


data=data.fillna(data['city_match'].value_counts().index[0])


# In[5]:


data['city_match'].unique()


# In[67]:


train=data[data.columns.difference(['apply','search_date_pacific','u_id','mgoc_id'])]


# In[ ]:


from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='Nan',strategy='mean')
i=imp.fit(train)
data_new=i.transform(train)


# In[69]:


X=data_new
Y=data['apply']


# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[89]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=23)


# In[73]:


train.head()


# In[75]:


fig,ax=plt.subplots(figsize=(12,6))
sns.scatterplot(x='main_query_tfidf',y='query_jl_score',data=train)


# In[79]:


corr = train.corr()
corr.style.background_gradient()


# In[93]:


from sklearn.feature_selection import mutual_info_classif,chi2,f_classif


# In[90]:


rewr=mutual_info_classif(X_train,y_train)


# In[91]:


rewr


# In[100]:


importances = pd.DataFrame({'feature':train.columns,'importance':np.round(rewr,5)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances


# In[110]:


e=f_classif(X_train,y_train)


# In[111]:


e


# In[115]:


de=pd.DataFrame({'feature':train.columns,'importance':e[0]})
de.sort_values(by='importance',ascending=False).set_index('feature')


# In[118]:


from sklearn.preprocessing import Normalizer,MinMaxScaler,StandardScaler,Imputer


# In[120]:


# lr=LogisticRegression()
# model=lr.fit(X_train,y_train)
pred=model.predict(X_test)


# In[6]:


data.head()


# In[121]:


model.intercept_


# In[124]:


model.coef_[0]


# In[126]:


coe=pd.DataFrame({'feature':train.columns,'coef':model.coef_[0]})
coe.sort_values(by='coef',ascending=False).set_index('feature')


# In[8]:


import math
a=math.exp(-2.501+7.050916*0.017)
b=1+1
p=a/b
p


# In[48]:


sns.distplot(num['main_query_tfidf'],kde=False)


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(data['main_query_tfidf'],kde=False)


# In[43]:


nume=[]
bina=[]
for i in data.columns[0:-4]:
    if data[i].value_counts().count()>2:
        nume.append(i)
    else:
        bina.append(i)


# In[44]:


# num


# In[45]:


from sklearn.preprocessing import MinMaxScaler,Imputer,Normalizer
i=Imputer()
num=i.fit_transform(data[nume])
m=Normalizer()
num=m.fit_transform(num)


# In[47]:


num=pd.DataFrame(num,columns=
['title_proximity_tfidf',
 'description_proximity_tfidf',
 'main_query_tfidf',
 'query_jl_score',
 'query_title_score',
 'job_age_days'])


# In[39]:


from sklearn.feature_selection import chi2,mutual_info_classif

for i in bina:
    data.fillna(data[i].value_counts().index[0],inplace=True)
chi=chi2(data[bina],data['apply'])
chi
mutual_info_classif()


# In[129]:


plt.hist(data['main_query_tfidf'])[0]


# In[130]:


import scipy.stats as stats


# In[133]:


stats.probplot(data['main_query_tfidf'],plot=plt)
plt.show()


# In[136]:


d=data['main_query_tfidf']
d=np.array(d)
d=d.reshape(-1,1)


# In[156]:


# q=Normalizer()
# e=q.fit(train)
# a=e.transform(train)
q=MinMaxScaler()
e=q.fit(train)
a=e.transform(train)


# In[149]:


stats.probplot(a['main_query_tfidf'],plot=plt)
plt.show()


# In[157]:


a=pd.DataFrame(a,columns=train.columns)


# In[158]:


from scipy import stats
stats.probplot(a['main_query_tfidf'],plot=plt)
plt.show()


# In[153]:


train.describe()


# In[154]:


stats.probplot(data['description_proximity_tfidf'],plot=plt)
plt.show()


# # 
# barplot
# countplot
# kdeplot- skewness/distribution
# histogram- skewness/distribution
# qq plot- probability plot to check for normality
# boxplot- used to identify outliers
# violin plot- used to find noise in data which can't be onbserved with box
# scatterplot- to identify relationship betn numerical data

# In[8]:


new=pd.read_csv('C:/Users/Akshat/Downloads/binary.csv')


# In[160]:


new.head()


# In[161]:


new.shape


# In[162]:


new.describe()


# In[164]:


new.isnull().sum()


# In[9]:


new[new.duplicated(keep=False)]


# In[169]:


new.loc[(new.gre == 700) & (new['gpa']==4.00) & (new['rank']==1)&(new.admit==1)]


# In[10]:


new.drop_duplicates(inplace=True)


# In[171]:


new[new.duplicated(keep=False)]


# In[172]:


new['rank'].unique()


# In[173]:


new['rank'].value_counts()


# In[174]:


new.admit.value_counts()


# In[175]:


mean_gre=new['gre'].mean()


# In[178]:


new['gre'].fillna(mean_gre,inplace=True)


# In[179]:


new['admit'].fillna(new.admit.value_counts().index[0],inplace=True)


# In[181]:


new=new.dropna(axis=0,how='all')


# In[11]:


corrr=new.corr()
corrr.style.background_gradient()


# In[201]:


fig,ax=plt.subplots(figsize=(8,6),)
#sns.barplot(x=new['rank'],y=new.gre,data=new,hue=new.admit,legend=False)
sns.catplot(x='rank',y='gre',data=new,hue='admit',legend=False,kind='bar',ax=ax)
plt.legend(loc='upper right')


# In[187]:


sns.countplot(new['rank'],data=new)


# In[198]:


sns.kdeplot(new.gre)


# In[199]:


plt.hist(new.gre)


# In[200]:


sns.boxplot(x=new['rank'],y=new.gre,data=new)


# In[202]:


sns.scatterplot(x=new.gre,y=new.gpa,data=new)


# In[205]:


from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
k=KFold(n_splits=10,random_state=9)
result=cross_val_score(lr,X_train,y_train,cv=k,scoring='roc_auc')


# In[206]:


result.mean()


# In[208]:


new.dtypes


# In[209]:


data.dtypes


# In[211]:


data['apply'].value_counts().count()


# In[214]:


bina=[]
num=[]
for i in data.columns[0:7]:
    if data[i].value_counts().count()==2:
        bina.append(i)
    else:
        num.append(i)


# In[215]:


bina


# In[216]:


num


# In[12]:


from sklearn.feature_selection import  f_classif,chi2,mutual_info_classif


# In[13]:


new.head()


# In[15]:


X=new[new.columns.difference(['admit'])]
Y=new['admit']


# In[30]:


anova=f_classif(X,Y)
mutual=mutual_info_classif(X,Y)


# In[31]:


mutual


# In[32]:


anova1=pd.DataFrame({'features':X.columns,'F-stat':anova[0],'p-values':anova[1]})
anova1=anova1.sort_values(by='F-stat',ascending=False).set_index('features')
mutual1=pd.DataFrame({'features':X.columns,'m-val':mutual}).sort_values(by='m-val').set_index('features')


# In[23]:


import seaborn as sns


# In[25]:


anova1


# In[29]:


sns.barplot(x=anova1['F-stat'],y=anova1.index,data=anova1,orient="h")


# In[35]:


sns.barplot(x=mutual1['m-val'],y=mutual1.index)


# In[34]:


mutual


# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[50]:


data.head(1)


# In[51]:


num.head()


# In[52]:


num['title_proximity_tfidf']=np.log(num['title_proximity_tfidf'])


# In[53]:


num.head()


# In[54]:


sns.kdeplot(num['title_proximity_tfidf'])


# In[ ]:


num['description_proximity_tfidf']=np.log(num['description_proximity_tfidf'])


# In[55]:


from sklearn.linear_model import LinearRegression


# In[56]:


li=LinearRegression()


# In[59]:


data.head()


# In[3]:


data.loc[data['job_age_days']>15]


# In[10]:


sns.barplot(y=data['description_proximity_tfidf'],x=data['city_match'],data=data.loc[data['job_age_days']>15])


# In[ ]:


sns

