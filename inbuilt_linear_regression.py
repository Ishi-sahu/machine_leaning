from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np
data=load_diabetes()
x=data.data
y=data.target
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y)
model=LinearRegression()
model_equation=model.fit(train_x,train_y)
output=model_equation.predict(test_x)
plt.scatter(test_y,output)
plt.show()