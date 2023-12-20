import os


import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import mean_squared_error


from pyrcn.echo_state_network import ESNRegressor

#Save dataset in dataArray

dataArray = []
for filename in os.listdir("Train"):
    fname = "Train/" + filename
    with open(fname) as f:
        for line in f: # read rest of lines
                dataArray.append([float(x) for x in line.split()])



dataArray = np.array(dataArray)
trainarrayx = dataArray[:,:3]
trainarrayy =  dataArray[:,3:]


dataArray2 = []
for filename in os.listdir("Test"):
    fname = "Test/" + filename
    with open(fname) as f:
        for line in f: # read rest of lines
                dataArray2.append([float(x) for x in line.split()])

dataArray2 = np.array(dataArray2)
testarrayx = dataArray2[:,:3]
testarrayy =  dataArray2[:,3:]


reg = ESNRegressor(spectral_radius = 0.99,sparsity = 0.5, hidden_layer_size = 1000)
reg.fit(X=trainarrayx, y=trainarrayy)


y_pred = reg.predict(testarrayx)  # output is the prediction for each input example

datafile_path = "output/prediction.dat" 
np.savetxt(datafile_path , np.array(y_pred), fmt=['%10.7f'])

th = []
th2 = []
for i in range(0,len(testarrayx)):
    th.append(2)
    th2.append(-1)


print("Schmitt trigger:")


mse = mean_squared_error(testarrayy,y_pred[:,0])
print("MSE for dataset "": " + str(mse))




############## 

plt.figure(0)

plt.plot(testarrayx[:,0],testarrayx[:,1], color = 'green')
plt.plot(testarrayx[:,0],y_pred, color='blue')
plt.plot(testarrayx[:,0],th, color='red', linestyle = 'dotted')
plt.plot(testarrayx[:,0],th2, color='red', linestyle = 'dotted')
plt.legend(["input", "output", "threshold"], loc ="lower left") 
txt="Prediction"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

############## 

plt.figure(1)


plt.plot(testarrayx[:,0],testarrayx[:,1], color = 'green')
plt.plot(testarrayx[:,0],th, color='red', linestyle = 'dotted')
plt.plot(testarrayx[:,0],th2, color='red', linestyle = 'dotted')
plt.plot(testarrayx[:,0],testarrayy, color='blue')
txt="True values"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)


