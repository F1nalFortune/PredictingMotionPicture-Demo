import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('./templates/index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    
    possible_scores = {
        0: '< $1 Million',
        1: '$1-20 Million',
        2: '$20-60 Million',
        3: '$60+ Million'
    }

    return render_template('index.html', prediction_text='Predicted Return: {}'.format(possible_scores[output]))


if __name__ == "__main__":
    app.run(debug=True)