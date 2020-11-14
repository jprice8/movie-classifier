import os
import time

import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

import pyprind
import pickle


def tokenizer(text):
    """
    Use regex to remove HTML and clean the text data and join with emoticons.
    Remove stop-words and tokenize the text.

    Returns:
        List of tokenized words from text input.
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
        + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stopwords.words('english')]
    return tokenized


def stream_docs(path):
    """
    A generator function that allows us to read-in the movie review, and label,
    one line at a time. Call the function like you would any Python generator
    with next(stream_docs(path="movie_data.csv"))
    """
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


"""
We're going to use scikit-learn's Hashing Vectorizer for OOC learning because
it does not require the script to hold the entire set of vocab in memory.
"""

# Instantiate our feature vectorizer and ML algo

vect = HashingVectorizer(decode_error='ignore',
    n_features=2**21,
    preprocessor=None,
    tokenizer=tokenizer)

"""
Loss log means we're running a logistic regression model.
"""

clf = SGDClassifier(loss='log', random_state=1, max_iter=1)

doc_stream = stream_docs(path='movie_data.csv')

# Now we can train our model
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])

for _ in range(45):
    x_train, y_train = get_minibatch(doc_stream, size=1000) # 45,000 reviews
    if not x_train:
        break
    x_train = vect.transform(x_train)
    clf.partial_fit(x_train, y_train, classes=classes)
    pbar.update()

# Test our model on the remaining 5,000 reviews
x_test, y_test = get_minibatch(doc_stream, size=5000)
x_test = vect.transform(x_test)

print('Accuracy: {:.3f}'.format(clf.score(x_test, y_test)))
print()
print('-' * 20)
print('Using the last 5000 documents to finish training our model...\n')

clf = clf.partial_fit(x_test, y_test)

print('Done!\n')

# Here we serialize our model using the pickle module
print()
print("Pickling stopwords and classifiers\n")
time.sleep(3)

dest = 'pkl_objects'
if not os.path.exists(dest):
    print("Creating new directory for pickling")
    os.makedirs(dest)

pickle.dump(stopwords, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

print('Done!')
