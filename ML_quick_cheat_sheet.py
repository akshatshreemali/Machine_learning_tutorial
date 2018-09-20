import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# gets unique values in your column
print(data['user'].unique())
dataset[0].value_counts()

# check for null values in data
data.isnull().values.any()

# check for null values with count
df.isnull().sum(axis=1)

# check for number of nulls per column
print(df.isnull().sum())

#  dropping columns which has only Nan values
tables=tables.dropna(axis=1, how='all') 
tables.head()

# deleteting multiple columns
df.drop(df.columns[[1, 69]], axis=1, inplace=True)

# This will delete all the column from index 29 till 38
tables.drop(tables.columns[29:39],axis=1,inplace=True) 

print('number of nulls in the label:{}'.format(dataset[0].isnull().sum()))
print('number of nulls in the body:{}'.format(dataset[1].isnull().sum()))

# displays if there's null value in any column
pd.isnull(data).sum()>0

# appending one df to other dframe
data=dta1.append(dta2)

# updating data based on keys
sample.loc[7:,'month'] = 4

# split column inside dataframe using split function
df1 = pd.DataFrame(df.Domicile.str.split(':',1).tolist(),
                                   columns = ['junk','NAIC'])

# check for duplicate values in data
data[data.duplicated(keep=False)]

# verifying duplicate data eg:
data.loc[(data.age == 19) & (data['fnlwgt']==251579)]

# dropping duplicate values
data.drop_duplicates(subset=['age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'salary'],inplace=True)
	   
	   
# get correlations from dataframe
corr = train.corr()
corr.style.background_gradient()

# converting text to number
data['sex']=np.where(data['sex']==' Male',0,1)

# updating columns
questions.loc[questions['choose_one']=='Relevant', 'choose_one_gold'] = 1
questions.loc[questions['choose_one']=='Not Relevant', 'choose_one_gold'] = 2
questions.loc[questions['choose_one']=="Can't Decide", 'choose_one_gold'] = 0

# using date column
data['joined_timestamp'] = pd.to_datetime(data['joined_timestamp'])
data['Year']=(data['joined_timestamp']).dt.year
data['Hour'] = pd.to_datetime(data['joined_timestamp']).dt.hour
data['Month']= pd.to_datetime(data['joined_timestamp']).dt.month


# Handling missing values
data=data.fillna(data['city_match'].value_counts().index[0])  # only categorical

from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='Nan',strategy='mean')
i=imp.fit(train)
data_new=i.transform(train)


# getting time of the day from date
data.loc[(data.Hour <= 3) | (data['Hour']>=23), 'Time_of_Day' ] = 'Late Night'
data.loc[(data.Hour <= 6) & (data['Hour']>3), 'Time_of_Day' ] = 'Early Morning'
data.loc[(data.Hour > 6) & (data['Hour']<=12), 'Time_of_Day' ] = 'Morning'
data.loc[(data.Hour > 12) & (data['Hour']<=16), 'Time_of_Day' ] = 'Afternoon'
data.loc[(data.Hour > 16) & (data['Hour']<=19), 'Time_of_Day' ] = 'Evening'
data.loc[(data.Hour >19) & (data['Hour']<=22), 'Time_of_Day' ] = 'Night'

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="Time_of_Day", y="num_auctions_engaged", data=data);

# check for which month
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(y='Month',data=data,order = data['Month'].value_counts().index)

# countplot
fig, ax = plt.subplots(figsize=(20, 8))
sns.countplot('Complaint_Type',data=tables)


# which day of the week
data['day_of_week'] = data['joined_timestamp'].dt.weekday_name
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(y='day_of_week',data=data,order = data['day_of_week'].value_counts().index)


# display kde plot
sns.kdeplot(data['gre'])

## Checking the shape of our continuous variable 'num_auctions_engaged '
fig, ax = plt.subplots(figsize=(12, 6))
sns.distplot(data['num_auctions_engaged'],bins=20,hist=False)

# boxplot
sns.boxplot(x=data['admit'],y=data['gre'],data=data)

#barplot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="user", y="num_auctions_engaged", data=data);

# barplot with respect to other categorical variable
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="Month", y="num_auctions_engaged",hue='user', data=data);


# scatterplot
sns.scatterplot(x=new.gre,y=new.gpa,data=new)



# split between output and others
X=data[data.columns.difference(['admit'])]
Y=data['admit']


# Separate categorical and continuous
bina=[]
num=[]
for i in data.columns[0:7]:
    if data[i].value_counts().count()==2:
        bina.append(i)
    else:
        num.append(i)

# Feature Selection
# Mutual Information
from sklearn.feature_selection import mutual_info_classif,chi2,f_classif
rewr=mutual_info_classif(X_train,y_train)
importances = pd.DataFrame({'feature':train.columns,'importance':np.round(rewr,5)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

# Anova
rewr=f_classif(X_train,y_train)
importances = pd.DataFrame({'feature':train.columns,'importance':np.round(rewr[0],5)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

# Chi-square





# string to categorical
for feature in combined_set.columns: # Loop through all columns in the dataframe
    if combined_set[feature].dtype == 'object': # Only apply for columns with categorical strings
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes # Replace strings with an integer


# ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics


def train_test(test_size):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state=25)
    return X_train, X_test, y_train, y_test
	
X_train, X_test, y_train, y_test=train_test(test_size=0.3)

def fit_model(model,X_train, X_test, y_train, y_test):
    ml_model=model
    #X_train, X_test, Y_train, Y_test=train_test()
    result=ml_model.fit(X_train,Y_train)
    prediction=result.predict(X_test)
    return prediction,result
	
prediction,result=fit_model(LogisticRegression(),X_train, X_test, y_train, y_test)

# using feature importances in RandomForestClassifier
result.feature_importances_
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(result.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head()
importances.plot.bar()


from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
k=KFold(n_splits=10,random_state=9)
result=cross_val_score(lr,X_train,y_train,cv=k,scoring='roc_auc')

def cross_validation(model):
    from sklearn import model_selection
    kfold = model_selection.KFold(n_splits=10, random_state=7)
	modelCV = model
	scoring = 'accuracy'
	results=(model_selection.cross_val_score(modelCV, X, Y, cv=kfold, scoring=scoring))
	a=(results.mean())
	return a
		
cross_validation(RandomForestClassifier())

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

# precision is  of the entire data set,87% of the recommended items were items that the user actually liked
# recall is measure of models completeness, of the entire data set, 89% of the users preferred items were recommended
#When we look over at recall we see that we get an 89. And what that is really saying is of all the products that users liked, 89% of those products were offered to them. 

def AUC(result,prediction):
    fpr, tpr, thresholds = roc_curve(y_test, result.predict_proba(X_test)[:,1])
    f_rf_roc_auc=roc_auc_score(y_test,prediction)
    plt.figure()
    plt.plot(fpr, tpr, label='Classification (area = %0.2f)' % f_rf_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
	
	
	
	
	
	
	

data.shape

data.isnull().sum()

data['abc'].unique()
data['abc'].value_counts()

data[data.duplicated(keep=False)]

data.drop_duplicates(inplace=True)

data.describe()

data.dropna(how='all')

data.fillna(data['abc'].value_counts().index[0])

data.drop(data['abc'],axis=1,inplace=True)

data['abc']=np.where(data['abc']=='male',0,1)


data.loc[data['abc']=='qwer','new_column']=1
data.loc[data['abc']=='wer','new_column']=2
data.loc[data['abc']=='qweqq','new_column']=0


c=data.corr()
c.style.background_gradient()

# EDA

sns.kdeplot(data['abc'])

sns.distplot(data['abc'],kde=False)

fig,ax=plt.subplots(figsize=(12,6))
sns.scatterplot(x,y)

sns.barplot(x,y,hue)

sns.boxplot
sns.violinplot(x,y)

from sklearn.feature_selection import chi2,mutual,f_classify
from sklearn.preprocessing import MinMaxScaler,Imputer,Normalizer


nume=[]
bina=[]
for i in data.columns:
  if data[i].value_counts().count()==2:
     bina.append(i)
  else:
     nume.append(i)

	 

X=data.columns.difference(data['test'])
Y=data['test']



chi,pval=chi2(X,Y)
mutual

imp=pd.DataFrame({'feature':

from sklearn.linear_model 
from sklearn.ensemble



