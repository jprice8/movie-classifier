from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

# cur dir = os.path.dirname(__file__)
# stop = pickle.load(open(
#     os.path.join(
#         'pkl_objects',
#         'stopwords.pkl'
#         ), 'rb'
#     )
# )
stopfile = os.path.join('pkl_objects', 'english.txt')

stop = []

with open(stopfile, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop.append(line.strip('\n'))


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
    tokenized = [w for w in text.split() if w not in stop] 
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
    n_features=2**21,
    preprocessor=None,
    tokenizer=tokenizer,
)
