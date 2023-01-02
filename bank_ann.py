# Artificial Neural Network

# Importing the libraries

import pandas as pd
import pickle
import tensorflow as tf

# Importing the dataset
dataset = pd.read_csv('csv/bank_ann.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer(
    [("Geography", OneHotEncoder(), [1]),
    ("Gender", OneHotEncoder(), [2])],
    remainder = 'passthrough')
X = ct.fit_transform(X)

X = X[:, 1:]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X,y,batch_size = 5, epochs = 5)

print(classifier.predict(584,'Spain','Female',	48,	2,	213146.2,	1,	1,	0,	75161.25))

# # Saving model to current directory
# pickle.dump(classifier, open("static/model_plk/model_bank.pkl", "wb"))
#
# # Loading model to compare results
# model = pickle.load(open("static/model_plk/model_bank.pkl", "rb"))
