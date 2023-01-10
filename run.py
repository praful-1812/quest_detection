import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

vectorizer_for_text = pickle.load(open("query_vectorizer.pickle", 'rb'))
classifier_for_text = pickle.load(open("query_classifier.pickle",'rb'))

ls  = ["net is working"]
text = vectorizer_for_text.transform(ls)
output = classifier_for_text.predict(text)

# print(output)
if output == [1]:
    print('question detected')
else:
    print("None")