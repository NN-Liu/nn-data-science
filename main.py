import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)



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
            model_car = pickle.load(open("model_car.pkl", "rb"))
            int_features = [float(x) for x in request.form.values()]
            final_features = [np.array(int_features)]
            prediction = model_car.predict(final_features)
        if request.method == "POST":
            output = round(prediction[0], 2)
        else:
            output = ""
        return render_template("car_regression.html", output= output)
    except:
        return "An error has incurred."


@app.route('/iris_classification', methods=["GET", "POST"])
def iris_classification():
    try:
            variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
            model_iris = pickle.load(open("model_iris.pkl", "rb"))
            float_features = [float(x) for x in request.form.values()]
            final_features = [np.array(float_features)]
        if request.method == "POST":
            prediction = variety_mappings[model_iris.predict(final_features)[0]]  # Retrieve from dictionary

        else:
            prediction = ""
        return render_template("iris_classification.html", prediction=prediction)
    except:
        return "An error has incurred."

@app.route('/bank_ann')
def bank_ann():
    return render_template("bank_ann.html")

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port = 5000)

