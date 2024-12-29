# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Naneshvaran 
RegisterNumber:  24900972
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head()

df.tail()
print(df.head())
print(df.tail())

X=df.iloc[:,:-1].values
print(X)

Y=df.iloc[:,1].values
print(Y)

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
print(Y_pred)

print(Y_test)

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![output(2) 1](https://github.com/user-attachments/assets/29b9a8fa-4a3b-4ed5-8ae8-ff924a59d662)
![output(2) 2](https://github.com/user-attachments/assets/0debea04-5f21-492d-b5bc-e6fdbb10ed14)
![output(2) 3](https://github.com/user-attachments/assets/f0997624-c211-4627-bf0c-28f5c5876695)
![output(2) 4](https://github.com/user-attachments/assets/59d9a1b9-a784-4f92-8298-59552668a8ba)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
