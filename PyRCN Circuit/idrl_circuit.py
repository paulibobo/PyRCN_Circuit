import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path
import numpy as np

from sklearn.metrics import mean_squared_error


from pyrcn.echo_state_network import ESNRegressor

#Save training dataset in dataArray

dataArray = []
i=0
k = 0
for filename in os.listdir("Train"):
    fname = "Train/" + filename
    with open(fname) as f:
        for line in f: # read rest of lines
                dataArray.append([float(x) for x in line.split()])
                k += 1
        i+=1

dataArray = np.array(dataArray)
trainarrayx = dataArray[:,:2]
trainarrayy =  dataArray[:,2:]

#Save test dataset in dataArray2
dataArray2 = []
i=0
k = 0
for filename in os.listdir("Test"):
    fname = "Test/" + filename
    with open(fname) as f:
        for line in f: # read rest of lines
                dataArray2.append([float(x) for x in line.split()])
                k += 1
        i+=1

dataArray2 = np.array(dataArray2)
testarrayx = dataArray2[:,:2]
testarrayy =  dataArray2[:,2:]

reg = ESNRegressor(spectral_radius = 0.99,sparsity = 0.3) #train model
reg.fit(X=trainarrayx, y=trainarrayy)


y_pred = reg.predict(testarrayx[0:5001])  # output is the prediction for each input example



datafile_path = "output/prediction.dat" 
np.savetxt(datafile_path , np.array(y_pred), fmt=['%10.7f','%10.7f','%10.7f'])

############## 

plt.figure(0)

plt.plot(testarrayx[0:5001,0],testarrayx[0:5001,1], color = 'green')

plt.plot(testarrayx[0:5001,0],y_pred[:,0], color='red')
plt.plot(testarrayx[0:5001,0],y_pred[:,1], color='blue')
plt.plot(testarrayx[0:5001,0],y_pred[:,2], color='orange')
plt.legend(["input", "output 1", "output 2","output 3"], loc ="center right") 
txt="Prediction"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

############## 

plt.figure(1)

plt.plot(testarrayx[0:5001,0],testarrayx[0:5001,1], color = 'green')
plt.plot(testarrayx[0:5001,0],testarrayy[0:5001,0], color='red')
plt.plot(testarrayx[0:5001,0],testarrayy[0:5001,1], color='blue')
plt.plot(testarrayx[0:5001,0],testarrayy[0:5001,2], color='orange')
plt.legend(["input", "output 1", "output 2","output 3"], loc ="center right") 
txt="True values"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)


