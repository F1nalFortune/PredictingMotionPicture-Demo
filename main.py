import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle


app = Flask(__name__)
model = pickle.load(open('lgbmClassifier.pkl', 'rb'))

possible_scores = {
    0: '< $1 Million',
    1: '$1-20 Million',
    2: '$20-60 Million',
    3: '$60+ Million'
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Predicted Return: {}'.format(possible_scores[output]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]

    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
