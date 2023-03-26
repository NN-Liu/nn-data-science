import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/car_regression', methods=["GET", "POST"])
def car_regression():
    try:
        dataset = pd.read_csv('csv/used_honda_price.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 2].values

        # Fitting Simple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X, y)

        # Saving model to current directory
        pickle.dump(regressor, open("model_car.pkl", "wb"))

        if request.method == "POST":
            model_car = pickle.load(open("model_car.pkl", "rb"))
            int_features = [float(x) for x in request.form.values()]
            final_features = [np.array(int_features)]
            prediction = model_car.predict(final_features)
            output = round(prediction[0], 2)
        else:
            output = ""
        return render_template("car_regression.html", output=output)
    except:
        return "An error has incurred."


@app.route('/iris_classification', methods=["GET", "POST"])
def iris_classification():
    try:
        # Importing the dataset
        dataset = pd.read_csv('csv/iris.csv')

        # Mapping varieties to dictionary
        variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        dataset = dataset.replace(['Setosa', 'Versicolor', 'Virginica'], [0, 1, 2])
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        # Fitting Kernel SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel='rbf', random_state=1)
        classifier.fit(X, y)

        # Saving model to current directory
        pickle.dump(classifier, open("model_iris.pkl", "wb"))

        if request.method == "POST":
            model_iris = pickle.load(open("model_iris.pkl", "rb"))
            float_features = [float(x) for x in request.form.values()]
            final_features = [np.array(float_features)]
            prediction = variety_mappings[model_iris.predict(final_features)[0]]  # Retrieve from dictionary
        else:
            prediction = ""
        return render_template("iris_classification.html", prediction=prediction)
    except:
        return "An error has incurred."


@app.route('/bank_ann')
def bank_ann():
    return render_template("bank_ann.html")


@app.route('/image_cnn')
def image_cnn():
    return render_template("image_cnn.html")


@app.route('/stock_price_rnn')
def stock_price_rnn():
    return render_template("stock_price_rnn.html")


@app.route('/nlp')
def nlp():
    return render_template("nlp.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
