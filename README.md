# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
DEVOLOPED BY : SREE NIVEDITAA SARAVANAN
REGISTER NUMBER : 212223230213
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/exp2ml.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')
```

## Output:
![image](https://github.com/sreeniveditaa/Find-the-best-fit-line-using-Least-Squares-Method/assets/147473268/c419b940-00a9-4e8d-8f2f-f50b267967d7)



## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
