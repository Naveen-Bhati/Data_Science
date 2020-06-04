import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt  = pd.read_csv("C:/Users/Albus Dumbledore/OneDrive/Desktop/DEsktop/machine learning/P14-Part7-Natural-Language-Processing/P14-Part7-Natural-Language-Processing/Section 33 - Natural Language Processing/Python/Restaurant_Reviews.tsv", delimiter = '\t', quoting =3)
'''
above we have used tsv(tab seperated values) because csv filed isnt good...delimeiter will
specify that tab...and quoting is used to ignore ""...'''

#cleaning of the text
import re 
import nltk
nltk.download('stopwords')  #this stopwords file contains the words  that are not relevant in reviews 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #this is used to change loved = loving  = love


corpus = []
for i in range(0,1000):
    reviews = re.sub('[^a-zA-Z]', ' ', dt['Review'][i])  # '^'  this sign is used not to remove alphabets
    reviews  = reviews.lower()  #to lower all alphabets
    reviews = reviews.split()
    ps = PorterStemmer()
    reviews = [ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))] # set is used only to increse speed if we have a large review
    reviews = ' '.join(reviews)
    corpus.append(reviews)
    
#bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #we also can use stopwords parameter in countervectorizer and other functions
#max_feature will remove non_relavant word and change features from 1565 to 1500

X = cv.fit_transform(corpus).toarray()
y = dt.iloc[:,1].values

#fit the model naive bayes
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state =0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix 
cm  = confusion_matrix(y_test,y_pred)

#the accuracy will be (55+91)/200 = 73  percentage