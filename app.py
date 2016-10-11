"""
To access server use:
$ ssh -l [username] env9ds4-l.ca.firstrain.net
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics

import pandas as pd
import numpy as np
import cPickle as pickle
import time
import dill
import operator
import math
import os

from flask import Flask, jsonify, request, Response

app = Flask(__name__)

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
            'message': 'Required JSON parameters: '+parameter
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

@app.route('/api/trainModel/', methods = ['POST'])
def api_trainmodel():

    if request.headers['Content-Type'] == 'application/json':
        data = request.json
        if 'features' not in data or 'target' not in data:
            return required_param('\'features\', \'target\'')
    else:
        return incorrect_type()

    numSamples = len(data['features'])

    if 'model' in data:
        classifier = models[data['model']]

    elif numSamples > 5000:
        classifier = LinearSVC(penalty='l1', dual=False)
    else:
        classifier = SVC(C=100, cache_size=500, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)

    count_vect, tfidf_transformer, clf, output = trainModel(data, classifier)

    if 'name' in data:
        print "Saving Model....."
        saveModel(data['name'], count_vect, tfidf_transformer, clf)

    return jsonify(output)

def relevancyScore(method, string):

    bagOfWords = count_vectorizers[method].transform([string])
    tfidf_vect = tfidf_transformers[method].transform(bagOfWords)

    output = {}
    classification = {1.0:'Positive', 0.0:'Negative'}
    output['prediction'] = classification[classifiers[method].predict(tfidf_vect)[0]]

    try:
        output['score'] = classifiers[method].decision_function(tfidf_vect)[0]
    except:
        output['score'] = classifiers[method].predict_proba(tfidf_vect)[:,1][0]

    if method in scorers:
        output['relevancy Score'], output['irrelevancy Score'] = scorers[method](output['score'])

    return output

def saveModel(method,count_vect, tfidf_transformer, clf):

    method = str(method)

    if not os.path.exists(method):
        os.makedirs(method)

    with open(method+'/count_vect', 'wb') as f:
        pickle.dump(count_vect, f)

    with open(method+'/model', 'wb') as f:
        pickle.dump(clf, f)

    with open(method+'/tfidf_transformer', 'wb') as f:
        pickle.dump(tfidf_transformer, f)

def loadModels(api_methods):

    classifiers,count_vectorizers,tfidf_transformers,scorers = dict(), dict(), dict(), dict()

    for method in api_methods:
        with open(method+'/count_vect', 'rb') as f:
            count_vectorizers[method] = pickle.load(f)

        with open(method+'/model', 'rb') as f:
            classifiers[method] = pickle.load(f)

        with open(method+'/tfidf_transformer', 'rb') as f:
            tfidf_transformers[method] = pickle.load(f)

        try:
            with open(method+'/scorer', 'rb') as f:
                scorers[method] = dill.load(f)
        except:
            continue

    return classifiers,count_vectorizers,tfidf_transformers,scorers

def trainModel(data, classifier):

    target = np.array(data['target'])
    text = data['features']

    print "Feature Transformation....."
    count_vect = CountVectorizer(ngram_range=(1,2), stop_words='english', decode_error='replace').fit(text)
    BOW_data = count_vect.transform(text)
    tfidf_transformer = TfidfTransformer(use_idf=True).fit(BOW_data)
    features = tfidf_transformer.transform(BOW_data)

    X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, random_state=42)

    print "Training Model....."
    clf = classifier.fit(X_train, y_train)
    cv_score = cross_val_score(clf, features, target, cv = 10)

    print "Making Predictions....."
    y_hat = clf.predict(X_test)

    try:
        y_scores = clf.decision_function(X_test)
    except:
        y_scores = clf.predict_proba(X_test)[:,1]

    output = dict()

    output['average accuracy'] = np.mean(cv_score)
    output['sample accuracies'] = cv_score.tolist()
    output['predictions'] = clf.predict(features).tolist()

    try:
        output['scores'] = clf.decision_function(features).tolist()
    except:
        output['scores'] = clf.predict_proba(features).tolist()

    try:
        output['feature coef'] = clf.coef_.tolist()
    except:
        output['feature coef'] = 'Not Available'

    output['Test:PR report'] = metrics.classification_report(y_test, y_hat, target_names=['Not Relevant', 'Relevant'])
    output['Test:true -: [marked - , marked +]'], output['Test:true +: [marked - , marked +]'] = metrics.confusion_matrix(y_test, y_hat).tolist()

    return count_vect, tfidf_transformer, clf, output

if __name__ == '__main__':

    api_methods = ['creditRelevancy','publisherClassifier']

    models = {'naive bayes': MultinomialNB(),
              'logistic regression':SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5,random_state=42),
              'random forest':RandomForestClassifier(n_estimators=10)}

    classifiers,count_vectorizers,tfidf_transformers,scorers = loadModels(api_methods)

    app.run(host='0.0.0.0', threaded=True)
