import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt  = pd.read_csv("C:/Users/Albus Dumbledore/OneDrive/Desktop/breast-cancer-wisconsin-data/Breast_cancer_dataset.csv")
X = dt.iloc[:,2:-1].values
y = dt.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state  =0, test_size = 0.2)



#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


sc_X_train = sc.fit_transform(X_train)
sc_X_test = sc.transform(X_test)

#fitting naive bayes model

from sklearn.naive_bayes import GaussianNB
classifier  = GaussianNB()
classifier.fit(sc_X_train,y_train)

y_pred = classifier.predict(sc_X_test)


from sklearn.metrics import confusion_matrix as cm
cmm  = cm(y_test,y_pred)

count = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        count+=1
accuracy = count/len(y_test)

