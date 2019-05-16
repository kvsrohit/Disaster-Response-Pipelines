import sys
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from sqlalchemy import create_engine

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, classification_report
from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import workspace_utils
from workspace_utils import active_session


def load_data(database_filepath):
    """Loads the sqllite database file and returns X,Y vectors and category names
    Args: 
    database_filepath: sqllite database file path
    Returns:
    X: input messages
    Y: target category classifications
    labels: category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    labels = Y.columns.values
    return X, Y, labels

def tokenize(text):
    """Processes the inputs by normalizing, tokenizing and converting to word stems
    Args: 
    text: text to be processes
    Returns:
    tokens: the processed and tokenized list of words
    """
    
    # Remove urls from text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # convert to lowe-case and remove any special characters.
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    words = word_tokenize(text)
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    # filter out the stop-words and lemmatize
    tokens = [lemmatizer.lemmatize(word.strip()) for word in words if word not in stop_words]

    return tokens

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    parameters = {
        'vect__max_df': (0.5, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 30],
        'clf__estimator__min_samples_split': [2, 5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=-1)
    return cv

def calculate_scores(actuals, predicted, labels):
    scores = []
    for i,label in enumerate(labels):
        scores.append({
            'label': label,
            'accuracy': accuracy_score(actuals[i], predicted[i]),
            'precision': precision_score(actuals[i], predicted[i], average='weighted'),
            'recall': recall_score(actuals[i], predicted[i], average='weighted'),
            'F1': f1_score(actuals[i], predicted[i], average='weighted'),
            'Fbeta': fbeta_score(actuals[i], predicted[i], beta=2, average='weighted')

        })
    scores = pd.DataFrame.from_dict(scores).set_index('label')
    return scores

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    scores = calculate_scores(Y_test.values, Y_pred, category_names)
    print(scores)

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as pickle_file:
        pickle.dump(model, pickle_file, fix_imports=True)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        with active_session():
            model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()