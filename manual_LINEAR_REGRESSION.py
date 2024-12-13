import numpy as np
data=np.loadtxt("data.csv",delimiter=",")
x=data[:,0].reshape(-1,1)  
y=data[:,1]
from sklearn import model_selection 
import pandas as pd
import matplotlib.pyplot as plt
train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y)

def fit(a,b):
   product=a*b
   xy_mean=np.mean(product)
   xmean=np.mean(a)
   ymean=np.mean(b)
   xmean_ymean=(xmean*ymean)
   xsqmean=np.mean(a*b)
   xmeanxmean=xmean*xmean
   num=xy_mean-xmean_ymean
   dem=xsqmean-xmeanxmean
   m=num/dem
   c=ymean-(m*xmean)
   return m,c

def predict(a,m,c):
   ypre=(m*a)+c
   return ypre

def score(ypre,ytest):
   u=np.sum(((ytest-ypre)**2))
   ymean=np.mean(ytest)
   v=np.sum(((ytest-ymean)**2))
   return (1-(u/v))

def cost(a,b,m,c):
   value=np.sum(((b-((m*a)+c))**2))
   return value

m,c=fit(train_x,train_y)
print("m:" ,m,",c: ",c)
pred=predict(test_x,m,c)
plt.scatter(test_y,pred)
plt.show()
ans=score(pred,test_y)
print("\n",ans)