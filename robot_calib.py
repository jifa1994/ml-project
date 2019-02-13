

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

import os
# print(os.getcwd())

names =[
    't',                                  # Time (secs)
    'q1', 'q2', 'q3',                     # Joint angle   (rads)
    'dq1', 'dq2', 'dq3',                  # Joint velocity (rads/sec)
    'I1', 'I2', 'I3',                     # Motor current (A)
    'eps21', 'eps22', 'eps31', 'eps32',   # Strain gauge measurements ($\mu$m /m )
    'ddq1', 'ddq2', 'ddq3'                # Joint accelerations (rad/sec^2)
]

df = pd.read_csv('/Users/puff/Pycharm/lab_robot_calib/exp1.csv',header=None,sep=',',names=names,index_col=0)

df.to_csv('Result.csv')

y = np.array(df['I2'])
t = np.array(df.index)

plt.plot(t,y)
plt.grid()

ytrain = np.array(df['I2'])
Xtrain = np.array(df[['q2','dq2','eps21','eps22','eps31','eps32','ddq2']])



from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(Xtrain,ytrain)

ytrain_pred = regr.predict(Xtrain)

plt.plot(t,ytrain)
plt.plot(t,ytrain_pred)
plt.legend(['actual','predicted'])
plt.show()

RSS_train = np.mean((ytrain-ytrain_pred)**2)/np.mean((ytrain-np.mean(ytrain))**2)
print(RSS_train)
'''
---------------------------------------------
 '''

df = pd.read_csv('exp2.csv',header=None,sep=',',names=names,index_col=0)


ytest = np.array(df['I2'])
Xtest = np.array(df[['q2','dq2','eps21','eps22','eps31','eps32','ddq2']])
ttest = np.array(df.index)
ytest_pred = regr.predict(Xtest)


plt.plot(t,ytest)
plt.plot(t,ytest_pred)
plt.legend(['actual','predicted'])

RSS_test = np.mean((ytest-ytest_pred)**2)/np.mean((ytest-np.mean(ytest))**2)

print(RSS_test)



