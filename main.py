import pickle
import pandas as pd
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import numpy as np
from datetime import datetime

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/car_regression', methods=["GET", "POST"])
def car_regression():
    try:
        if request.method == "POST":
            model_car = pickle.load(open("static/model_plk/model_car.pkl", "rb"))
            int_features = [float(x) for x in request.form.values()]
            final_features = [np.array(int_features)]
            prediction = model_car.predict(final_features)
            output = round(prediction[0], 2)
            return render_template("car_regression.html", output=f"The price for used Honda car is {output}")
        else:
            return render_template("car_regression.html", output="")
    except:
        return "An error has incurred."


@app.route('/iris_classification', methods=["GET", "POST"])
def iris_classification():
    try:
        variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        if request.method == "POST":
            model_iris = pickle.load(open("static/model_plk/model_iris.pkl", "rb"))
            float_features = [float(x) for x in request.form.values()]
            final_features = [np.array(float_features)]
            prediction = variety_mappings[model_iris.predict(final_features)[0]]  # Retrieve from dictionary
            print(prediction)
            return render_template("iris_classification.html", prediction=f"The flower is {prediction}")
        else:
            return render_template("iris_classification.html", prediction="")
    except:
        return "An error has incurred."

@app.route('/bank_ann')
def bank_ann():
    return render_template("bank_ann.html")

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port = 5000)
