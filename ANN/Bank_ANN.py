import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt = pd.read_csv("Churn_Modelling.csv")

X = dt.iloc[:,3:13].values
y = dt.iloc[:,13].values

#Encoding Categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer  #in new version we can use column transformer instead of lavelencoding

X_gender = LabelEncoder()
X[:,2] = X_gender.fit_transform(X[:,2]) #we will use label encoding for gender

columnTransformer_city = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough') #here 1 is the index number and encoder is just a name

X = np.array(columnTransformer_city.fit_transform(X), dtype = np.str)

X = X[:,1:] #drop the first dummy column

#split data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size =0.2,random_state = 0)

#feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)




#importing keras and modules
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN
classifier = Sequential()

#adding input layer and first hidden layer
classifier.add(Dense(output_dim = 6,input_dim =11,init = 'uniform',activation = 'relu'))

#adding 2nd hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))

#adding output layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer = 'adam', loss ='binary_crossentropy',metrics = ['accuracy']) #adam is a type of stocastic gradient decent

#fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size = 10,nb_epoch = 100) 

#pedicting the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #this will classify result in true false

#confusion matrix
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test,y_pred)
