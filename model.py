"""
Deploy the model as an API.
"""
from flask import Flask
from flask import request
import pickle
import pandas as pd


app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict_model():
    if request.method == 'POST':
        data = request.json  # a multidict containing POST data

        dict_value = {'EXT_SOURCE_3': data['EXT_SOURCE_3'],
                      'DAYS_WORKING_PER': data['DAYS_WORKING_PER'],
                      'MAX_DAYS_SOMETHING_CHANGED': data['MAX_DAYS_SOMETHING_CHANGED'],
                      'GOODS_PRICE_CREDIT_PER': data['GOODS_PRICE_CREDIT_PER'],
                      'DAYS_ID_PUBLISH': data['DAYS_ID_PUBLISH'],
                      'ANNUITY_DAYS_BIRTH_PERC': data['ANNUITY_DAYS_BIRTH_PERC'],
                      'AMT_GOODS_PRICE': data['AMT_GOODS_PRICE'],
                      'FLOORSMAX_MEDI': data['FLOORSMAX_MEDI']}

        X_new = pd.DataFrame(dict_value, index=[0])

        print('data received', data)
        print('dataframe created', X_new)
        file_name = "xgb_reg.pkl"

        # load the pickle
        model = pickle.load(open(file_name, "rb"))

        prediction = model.predict(X_new)

        return f"The prediction for this individual is {round(prediction[0], 2)}!"
