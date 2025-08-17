import numpy as np
from flask import Flask, jsonify, request
import pickle
import pandas as pd

# create flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return jsonify({'message': 'Welcome to the Diabetes Prediction API'})
    else:
        return jsonify({'message': 'POST method not supported on this endpoint'})
    

@app.route('/predict/', methods=['GET'])
def diabetes_predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        # Load model
        model = pickle.load(open('model_pmb.pkl', 'rb'))

        # Get parameters from query string
        age = request.args.get('age')
        gender = request.args.get('gender')
        bmi = request.args.get('bmi')
        bp = request.args.get('bp')
        s1 = request.args.get('s1')
        s2 = request.args.get('s2')
        s3 = request.args.get('s3')
        s4 = request.args.get('s4')
        s5 = request.args.get('s5')
        s6 = request.args.get('s6')

        # Convert inputs to float
        input_data = [float(age), float(gender), float(bmi), float(bp),
                      float(s1), float(s2), float(s3), float(s4), float(s5), float(s6)]

        # Create DataFrame with correct column names
        test_df = pd.DataFrame([input_data], columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])

        # Predict
        pred_level = model.predict(test_df)

        return jsonify({'Predicted disease progression': round(float(pred_level[0]), 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

# driver function
if __name__ == '__main__':
    app.run(debug=True)

