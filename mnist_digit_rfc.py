import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dt = pd.read_csv("C:/Users/Albus Dumbledore/OneDrive/Desktop/mnist-in-csv/mnist_train.csv")
dt2 = pd.read_csv("C:/Users/Albus Dumbledore/OneDrive/Desktop/mnist-in-csv/mnist_test.csv")
X_train = dt.iloc[:,1:].values
y_train = dt.iloc[:,0].values
X_test = dt2.iloc[:,1:].values
y_test = dt2.iloc[:,0].values


#Random forest classification
a = dt.iloc[3,1:].values
a= a.reshape(28,28)
plt.imshow(a)


from sklearn.ensemble import RandomForestClassifier as rfc
classifier = rfc(n_estimators = 10,random_state = 0) 
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

#comfusion matrics

from sklearn.metrics import confusion_matrix as cm
cmm = cm(y_test,y_pred)


count = 0

for i in range(len(y_pred)):
    
    if y_pred[i]==y_test[i]:
        count = count+1

accuracy = count/ len(y_pred)