import flask
from flask import request, jsonify
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
import pickle
import ast
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Create some test data for our catalog in the form of a list of dictionaries.


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''


@app.route('/api/v1/resources/books/all', methods=['GET'])
def api_all():
    return jsonify(books)


@app.route('/api/', methods=['GET'])
def api_id():
    output = {'text': '',
              'SVM': '',
              'Naive_Bayes': '',
              }
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'text' in request.args:
        input_text = str(request.args['text'])
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        def text_preprocessing(text):
            # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
            input_text = text.lower()

            # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
            text_words_list = word_tokenize(text)

            # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(text_words_list):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(
                        word, tag_map[tag[0]])
                    Final_words.append(word_Final)
                # The final processed set of words for each iteration will be stored in 'text_final'
            return str(Final_words)

        # Loading Label encoder
        labelencode = pickle.load(open('labelencoder_fitted.pkl', 'rb'))

        # Loading TF-IDF Vectorizer
        Tfidf_vect = pickle.load(open('Tfidf_vect_fitted.pkl', 'rb'))

        # Loading models
        SVM = pickle.load(open('svm_trained_model.sav', 'rb'))
        Naive = pickle.load(open('nb_trained_model.sav', 'rb'))

        # Inference
        sample_text = input_text
        sample_text_processed = text_preprocessing(sample_text)
        sample_text_processed_vectorized = Tfidf_vect.transform(
            [sample_text_processed])

        prediction_SVM = SVM.predict(sample_text_processed_vectorized)
        prediction_Naive = Naive.predict(sample_text_processed_vectorized)

        print("Prediction from SVM Model:",
              labelencode.inverse_transform(prediction_SVM)[0])
        print("Prediction from NB Model:",
              labelencode.inverse_transform(prediction_Naive)[0])
        output = {'text': input_text,
                  'SVM': labelencode.inverse_transform(prediction_SVM)[0],
                  'Naive_Bayes': labelencode.inverse_transform(prediction_Naive)[0],
                  }
        return output
    else:
        return output

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for book in books:
        if book['id'] == id:
            results.append(book)

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)


app.run()
