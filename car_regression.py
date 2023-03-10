# Simple Linear Regression

# Importing the libraries
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('csv/used_honda_price.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Saving model to current directory
pickle.dump(regressor, open("model_car.pkl", "wb"))

# Loading model to compare results
model = pickle.load(open("model_car.pkl", "rb"))

