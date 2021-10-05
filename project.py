import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sn

import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("diabetics-prediction-main/Diabetes_data.csv")

data.head()

data.info()

data.shape

data

data.columns

data.isnull()

data.isnull().sum()

data.duplicated().sum()

data.describe()

data.Insulin.quantile(0.99999)

data.loc[data['Insulin']>200.0,'Insulin']=np.mean(data['Insulin'])

data.shape

data.describe()

data.Outcome.value_counts()

data['Outcome'].value_counts(normalize=True)

data['Glucose'] = np.where(data['Glucose']==0, data['Glucose'].mean(), data['Glucose'])
data['BloodPressure'] = np.where(data['BloodPressure']==0, data['BloodPressure'].mean(), data['BloodPressure'])
data['SkinThickness'] = np.where(data['SkinThickness']==0, data['SkinThickness'].mean(), data['SkinThickness'])
data['Insulin'] = np.where(data['Insulin']==0, data['Insulin'].mean(), data['Insulin'])
data['BMI'] = np.where(data['BMI']==0, data['BMI'].mean(), data['BMI'])

plt.figure(figsize=(12,8))
sns.boxplot(data=data)
plt.show()

data.plot.scatter('Age','Insulin')

data.plot.scatter('Glucose','BloodPressure')

sn.distplot(data['Age'])

sn.displot(data['Glucose'])

data['Outcome'].value_counts().plot.bar()

data.groupby(['Outcome'])['Pregnancies'].mean()

data.groupby(['Outcome'])['Age'].mean()

data.groupby(['Outcome'])['Glucose'].mean()

data.groupby(['Outcome'])['Insulin'].mean()

data.groupby(['Outcome'])['BloodPressure'].mean()

data.groupby(['Outcome'])['BMI'].mean()

data.corr()

cor = data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(cor,annot=True,cmap='coolwarm')#if we will not write annot=True then the values will not show
plt.show()

from scipy.stats import ttest_rel

ttest_rel(data['Age'],data['Outcome'] ,nan_policy='omit')#paired t test

ttest_rel(data['Insulin'],data['Outcome'] ,nan_policy='omit')

# Independent and Dependent Feature:
x = data.iloc[:, :-1]
y = data.iloc[:, -1]



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print(x_train.shape)

print(x_test.shape)

## MODEL BUILDING


from sklearn.linear_model import LogisticRegression

ir=LogisticRegression()

ir.fit(x_train,y_train)

prediction=ir.predict(x_test)

prediction

from sklearn.metrics import accuracy_score,recall_score

accuracy_score(y_test,prediction)

recall_score(y_test,prediction)

## Descion tree


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=4, random_state=0)

clf.fit(x_train,y_train)

clf.score(x_train,y_train)

predict=clf.predict(x_test)

accuracy_score(y_test,predict)

## Knn



from sklearn.neighbors import KNeighborsClassifier
mode12 = KNeighborsClassifier(n_neighbors = 3)#no of neighbors is hpyer parameter


mode12.fit(x_train,y_train)

mode12.score(x_train,y_train)

pred=mode12.predict(x_test)

accuracy_score(pred,y_test)
import pickle

filename='diabetic.pkl'

pickle.dump(ir,open(filename,'wb'))
