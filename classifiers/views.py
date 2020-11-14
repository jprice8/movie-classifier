from django.shortcuts import render

from .forms import SentimentForm
from .models import Movie, Sentiment

from .vectorizer import vect

import os
import pickle
import numpy as np

classifier = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))


def classify(reviews):
    label = {0: 'negative', 1: 'positive'}
    x = vect.transform([reviews])
    pred = classifier.predict(x)[0]
    proba = np.max(classifier.predict_proba(x))
    return label[pred], proba


def train(review, y):
    x = vect.transform([review])
    classifier.partial_fit(x, [y])


def classify_review(request):
    form = SentimentForm(request.POST or None)
    if form.is_valid():
        reviews = form.cleaned_data.get('reviews')
        gender = form.cleaned_data.get('gender')
        age = form.cleaned_data.get('age')
        country = form.cleaned_data.get('country')
        movie = form.cleaned_data.get('movie')
        my_results, my_proba = classify(reviews)

        context = {
            'result': my_results,
            'probability': round(my_proba * 100, 2),
            'review': reviews,
            'gender': gender,
            'age': age,
            'country': country,
            'movie': movie,
        }
        return render(request, 'classifiers/prediction.html', context)
    context = {
        'form': form,
    }
    return render(request, 'classifiers/movie_classify.html', context)


def feedback(request):

    feedback = request.POST['feedback_button']
    review = request.POST['review']
    prediction = request.POST['result']
    gender = request.POST['gender']
    age = request.POST['age']
    country = request.POST['country']
    movie = request.POST['movie']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    my_movie = Movie.objects.filter(title=movie).first()
    sentiment = Sentiment(reviews=review, results=y, gender=gender, age=age, country=country,
    movie=my_movie)
    sentiment.save()
    context = {}
    return render(request, 'classifiers/thanks.html', context)
