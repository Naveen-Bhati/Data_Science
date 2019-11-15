
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
x=iris.data
y=iris.target
estimator = KNeighborsClassifier(n_neighbors=1)
estimator.fit(x,y)
print( estimator.predict([[3,4,5,2]]) )
