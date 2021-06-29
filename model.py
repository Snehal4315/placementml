import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split as tts

data = pd.read_csv('C:/Users/HP/Desktop/Snehal/PlacementML/Placement_Data_Full_Class.csv')

# data.head()

# sns.heatmap(data.isna())

#assuming that not placed student's salary is 0.
data['salary'] = data.salary.fillna(0)

# data.isna().sum()

# data.info()

# plt.subplot(3,2,1)
# sns.boxplot(data.ssc_p)
# plt.subplot(3,2,2)
# sns.boxplot(data.hsc_p)
# plt.subplot(3,2,3)
# sns.boxplot(data.degree_p)
# plt.subplot(3,2,4)
# sns.boxplot(data.etest_p)
# plt.subplot(3,2,5)
# sns.boxplot(data.mba_p)

clmns = ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status']
# clmns
from sklearn.preprocessing import LabelEncoder
lblen = LabelEncoder()
for clmn in clmns:    
    data[clmn] = lblen.fit_transform(data[clmn])
# data

train_data = data.drop('status',axis=1)
train_data = train_data.drop('salary',axis=1)
train_data = train_data.drop('sl_no',axis=1)
# print(train_data)

x_train,x_test,y_train,y_test = tts(train_data,data.status,test_size=0.2,random_state=42)

logreg = LogisticRegression()
logreg.fit(x_train,y_train)

linreg = LinearRegression()
linreg.fit(x_train,y_train)

pickle.dump(logreg,open('C:/Users/HP/Desktop/Snehal/PlacementML/model.pkl','wb'))
# pickle.dump(linreg,open('C:/Users/HP/Desktop/Snehal/PlacementML/model1.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
# modelreg = pickle.load(open('C:/Users/HP/Desktop/Snehal/PlacementML/model1.pkl','rb'))
print(model.predict([[1,67.00,1,91.00,1,1,58.00,2,0,55.0,1,58.80]]))
# print(modelreg.predict([[1,67.00,1,91.00,1,1,58.00,2,0,55.0,1,58.80]]))

# logreg_pred = logreg.predict(x_test)
# print(confusion_matrix(logreg_pred,y_test))
# print(accuracy_score(logreg_pred,y_test))