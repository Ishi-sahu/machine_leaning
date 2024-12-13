#using inbuit decision tree

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn .metrics import accuracy_score
import matplotlib.pyplot as plt
data=load_wine()
x=data.data
y=data.target
xtrain,xtest,ytrain,ytest=train_test_split(x,y)
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
y_pre=model.predict(xtest)
#plt.figure(figsize=(12,8))
plot_tree(model)
plt.show()
