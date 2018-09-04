# -*- coding: utf-8 -*-
from flask import Flask, request

import json
import logging
import os

from sklearn.externals import joblib

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(APP_ROOT, 'classifier.pkl')

PORT = 5000

app = Flask(__name__)
logging.basicConfig(filename='movie_classifier.log', level=logging.DEBUG)
model = joblib.load(MODEL)
label = {0:'negative', 1:'positive'}


@app.route('/')
def home():
    return 'It works.'


def predict(model, text):
    return label[model.predict([text])[0]]


@app.route('/review', methods=['GET'])
def extract():
    """Return the movie review sentiment score.
    
    Returns a JSON object :
    {
         "sentiment": "positive"
    }
    """
    if request.method == 'GET':
        description = request.args.get('text', '')
        
        result = {
            'sentiment': predict(model, description)
        }
        return json.dumps(result) 


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=PORT)

