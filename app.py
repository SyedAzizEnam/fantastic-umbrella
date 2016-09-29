from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pandas as pd
import cPickle as pickle
import time

from flask import Flask, jsonify, request, Response

app = Flask(__name__)

with open('selected_vocab', 'rb') as f:
    selected_vocab = pickle.load(f)

with open('SVM_rbf_9_28_2016', 'rb') as f:
    clf = pickle.load(f)

with open('tfidf_transformer', 'rb') as f:
    tfidf_transformer = pickle.load(f)

count_vect = CountVectorizer(ngram_range=(1,2), stop_words='english', vocabulary=selected_vocab)

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
def incorrect_use(parameter):
    message = {
            'status': 404,
            'message': 'Required JSON parameter: '+parameter
    }
    resp = jsonify(message)
    resp.status_code = 404

    return resp


@app.route('/api/getRelevancyScore/v1.0', methods=['POST'])
def api_relevancyscore():

    if request.headers['Content-Type'] == 'application/json':
        data = request.json
        if 'summary' not in data:
            return incorrect_use('summary')
    else:
        return incorrect_type()

    output = relevancyScore(data['summary'])

    return jsonify(output)


def relevancyScore(string):

    bagOfWords = count_vect.transform([string])
    tfidf_vect = tfidf_transformer.transform(bagOfWords)

    output = {}
    classification = {1.0:'Credit Relevant', 0.0:'Not Creadit Relevant'}
    output['prediction'] = classification[clf.predict(tfidf_vect)[0]]
    output['score'] = clf.decision_function(tfidf_vect)[0]

    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0')
