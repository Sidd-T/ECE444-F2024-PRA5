from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

@application.route('/', methods=['GET', 'POST']) 
def index():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    input_text = request.json.get('input_text')
    if input_text:
        output = load_model(input_text)
        return jsonify({'output': output})
    return jsonify({'error': 'No input text provided'}), 400

def load_model(str):
    loaded_model = None
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)
    
    vectorizer = None
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

    prediction = loaded_model.predict(vectorizer.transform([str]))[0]

    return prediction


if __name__ == "__main__":  
    application.run(port=5000, debug=True)
