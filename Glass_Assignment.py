# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 03:27:49 2021

@author: ASUS
"""

#Importing all the required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC

#Loading up the data file
file=pd.read_csv("D:\Data Science Assignments\R-Assignment\KNN_Classifier\glass.csv")
file.head()
file.shape
file.describe()

#Data Manipulation
def norm(i):
    x=((i-i.min())/(i.max()-i.min()))
    return x    
data=norm(file.iloc[:,0:9])
data['Type']=file.Type
data.head()

#Splitting the data file into training  and testing data 
train,test=train_test_split(data,test_size=0.20)
train.Type.value_counts()  #Checking the quality of the data distribution
test.Type.value_counts()

#Checking the model with k-value=3
n1=KNC(n_neighbors=3)
n1.fit(train.iloc[:,0:9],train.loc[:,'Type'])
train_acc1=np.mean(n1.predict(train.iloc[:,0:9])==train.loc[:,'Type'])
test_acc1=np.mean(n1.predict(test.iloc[:,0:9])==test.loc[:,'Type'])
train_acc1      #Training Accuracy
test_acc1       #Testing Accuracy

#Checking the model with k-value=4
n2=KNC(n_neighbors=4)
n2.fit(train.iloc[:,0:9],train.loc[:,'Type'])
train_acc2=np.mean(n2.predict(train.iloc[:,0:9])==train.loc[:,'Type'])
test_acc2=np.mean(n2.predict(test.iloc[:,0:9])==test.loc[:,'Type'])
train_acc2      #Training Accuracy
test_acc2       #Testing Accuracy

#Testing the model with a range of k-values
acc=[]
for i in range(1,20):
    n=KNC(n_neighbors=i)
    n.fit(train.iloc[:,0:9],train.loc[:,'Type'])
    train_acc=np.mean(n.predict(train.iloc[:,0:9])==train.loc[:,'Type'])
    test_acc=np.mean(n.predict(test.iloc[:,0:9])==test.loc[:,'Type'])
    acc.append([i,train_acc,test_acc])
accuracy=pd.DataFrame(acc)
accuracy.shape
accuracy.columns=["I_Values","Training_Accuracy","Testing_Accuracy"]
print(accuracy)


#Plotting Training Accuracy Values
plt.plot(range(1,20),accuracy.Training_Accuracy,"ro-",scalex=True)
plt.xlabel("KNeighbour Values")
plt.ylabel("Accuracy")
plt.title("Accuracy of model with Training Dataset ")


#Plotting the Testing accuracy values
plt.plot(range(1,20),accuracy.Testing_Accuracy,"go-",scalex=True)
plt.xlabel("KNeighbour Values")
plt.ylabel("Accuracy")
plt.title("Accuracy of model with Testing Dataset ")