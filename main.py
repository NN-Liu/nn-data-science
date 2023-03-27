import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/car_regression', methods=["GET"])
def car_regression():
    return render_template("car_regression.html")


@app.route('/car_regression_result', methods=["POST"])
def car_regression_result():
    try:
        dataset = pd.read_csv('csv/used_honda_price.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 2].values

        # Fitting Simple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X, y)

        # Saving model to current directory
        pickle_out = open("model_car.pickle", "wb")
        pickle.dump(regressor, pickle_out)
        pickle_out.close()

        model_car = pickle.load(open("model_car.pickle", "rb"))
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model_car.predict(final_features)
        output = round(prediction[0], 2)
        return render_template("car_regression_result.html", output=output)
    except:
        return "An error has incurred."


@app.route('/iris_classification', methods=["GET"])
def iris_classification():
    return render_template("iris_classification.html")


@app.route('/iris_classification_result', methods=["POST"])
def iris_classification_result():
    try:

        # Importing the dataset
        dataset = pd.read_csv('csv/iris.csv')

        # Mapping varieties to dictionary
        dataset = dataset.replace(['Setosa', 'Versicolor', 'Virginica'], [0, 1, 2])
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        # Fitting Kernel SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel='rbf', random_state=1)
        classifier.fit(X, y)

        # Saving model to current directory
        pickle_out = open("model_iris.pickle", "wb")
        pickle.dump(classifier, pickle_out)
        pickle_out.close()

        variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        model_iris = pickle.load(open("model_iris.pickle", "rb"))
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction = variety_mappings[model_iris.predict(final_features)[0]]  # Retrieve from dictionary

        return render_template("iris_classification_result.html", prediction=prediction)
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
    app.run(host='0.0.0.0', port=5000, debug=True)
