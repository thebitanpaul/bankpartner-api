from distutils.log import debug
from flask import Flask, jsonify , request
import pickle
import numpy as np
import pandas as pd
import sklearn as sd

model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)
@app.route('/')
def home():
    return "Good Job ML Engineer"


@app.route('/predict', methods=['POST'])
def predict():
    CreditScore = request.form.get('CreditScore')
    Age = request.form.get('Age')
    Tenure = request.form.get('Tenure')
    Balance = request.form.get('Balance')
    NumberOfProducts = request.form.get('NumberOfProducts')
    HasCrCard = request.form.get('HasCrCard')
    IsActiveMember = request.form.get('IsActiveMember')
    EstimatedSalary = request.form.get('EstimatedSalary')

    input_query = np.array(
        [[CreditScore, Age, Tenure, Balance, NumberOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
    result = model.predict(input_query)[0]

    return jsonify({'stay': str(result)})

if __name__ ==  "__main__":

    app.run()