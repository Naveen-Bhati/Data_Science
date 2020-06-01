import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt  = pd.read_csv("C:/Users/Albus Dumbledore/OneDrive/Desktop/machine learning/P14-Random-Forest-Regression/Random_Forest_Regression/Position_Salaries.csv")
X = dt.iloc[:, 1:2].values
y = dt.iloc[:, 2].values


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X,y)

y_pred = regressor.predict([[6.5]])

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#further we will take n_estimators =100,300 and then predict