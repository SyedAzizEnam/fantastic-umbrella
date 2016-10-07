"""
To access server use:
$ ssh -l [username] env9ds4-l.ca.firstrain.net
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pandas as pd
import cPickle as pickle
import time
import dill

from flask import Flask, jsonify, request, Response

app = Flask(__name__)

api_methods = ['creditRelevancy', 'companyRelevancy', 'sourceDetector']

classifiers,count_vectorizers,tfidf_transformers,scorers = dict(), dict(), dict(), dict()

for method in api_methods:
    with open(method+'/selected_vocab', 'rb') as f:
        vocab = pickle.load(f)
        count_vectorizers[method] = CountVectorizer(ngram_range=(1,2), stop_words='english', vocabulary=vocab)

    with open(method+'/model', 'rb') as f:
        classifiers[method] = pickle.load(f)

    with open(method+'/tfidf_transformer', 'rb') as f:
        tfidf_transformers[method] = pickle.load(f)

    try:
        with open(method+'/scorer', 'rb') as f:
            scorers[method] = dill.load(f)
    except:
        continue

@app.errorhandler(404)
def incorrect_type(error=None):
    message = {
            'status': 404,
            'message': 'Required Content-Type: application/json'
    }
    resp = jsonify(message)
    resp.status_code = 404

    return resp

@app.errorhandler(404)
def methodNotFound(method):
    message = {
            'status': 404,
            'message': 'Method '+method+' not found'
    }
    resp = jsonify(message)
    resp.status_code = 404

    return resp

@app.errorhandler(404)
def required_param(parameter):
    message = {
            'status': 404,
            'message': 'Required JSON parameter: '+parameter
    }
    resp = jsonify(message)
    resp.status_code = 404

    return resp

@app.errorhandler(404)
def incorrect_url(error=None):
    message = {
            'status': 404,
            'message': 'Correct Usage: /api/<method>/v1.0?title=title text&body=body text'
    }
    resp = jsonify(message)
    resp.status_code = 404

    return resp


@app.route('/api/<string:method>/v1.0', methods=['GET','POST'])
def api_relevancyscore(method):

    if method not in api_methods:
        return methodNotFound(method)

    if request.method == 'GET':
        if 'body' not in request.args:
            return incorrect_url()
        data = request.args

    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            data = request.json
            if 'body' not in data:
                return required_param('body')
        else:
            return incorrect_type()

    output = relevancyScore(method, data['body'])

    return jsonify(output)

def relevancyScore(method, string):

    bagOfWords = count_vectorizers[method].transform([string])
    tfidf_vect = tfidf_transformers[method].transform(bagOfWords)

    output = {}
    classification = {1.0:'Positive', 0.0:'Negative'}
    output['prediction'] = classification[classifiers[method].predict(tfidf_vect)[0]]

    try:
        output['score'] = classifiers[method].predict_proba(tfidf_vect)[:,1][0]
    except:
        output['score'] = classifiers[method].decision_function(tfidf_vect)[0]

    if method in scorers:
        output['relevancy Score'], output['irrelevancy Score'] = scorers[method](output['score'])

    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
