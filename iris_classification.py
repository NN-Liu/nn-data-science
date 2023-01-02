# Importing the libraries
import pickle
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('static/csv/iris.csv')

# Mapping varieties to dictionary
variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
dataset = dataset.replace(['Setosa','Versicolor','Virginica'],[0,1,2])
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1 ].values


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 1)
classifier.fit(X, y)

#Saving model to current directory
pickle.dump(classifier, open("static/model_plk/model_iris.pkl", "wb"))

# Loading model to compare results
model = pickle.load(open("static/model_plk/model_iris.pkl", "rb"))