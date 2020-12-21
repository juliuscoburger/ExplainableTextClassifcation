import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV


#import 20news dataset
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)



word_ngram_clf = Pipeline([
    ('vect', CountVectorizer(analyzer='word')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])
 
char_ngram_clf = Pipeline([
    ('vect', CountVectorizer(analyzer='char', ngram_range=(1,5))),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])


word_ngram_clf.fit(twenty_train.data, twenty_train.target)  
char_ngram_clf.fit(twenty_train.data, twenty_train.target)  

predicted_word = word_ngram_clf.predict(twenty_test.data)
predicted_char = char_ngram_clf.predict(twenty_test.data)



print("Word-ngarm:", np.mean(predicted_word == twenty_test.target))  
print("Char-ngram:", np.mean(predicted_char == twenty_test.target))  
