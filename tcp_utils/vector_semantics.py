import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator,TransformerMixin
from nltk.lm.preprocessing import flatten
from nltk.lm.vocabulary import Vocabulary


def process_text(text:pd.Series) -> str:
    #Assumes input is tokenized with tokens being separated by whitespace
    assert isinstance(text,pd.Series), 'Text must be a pandas series object!'
    return ' '.join(text.values)


def create_vocabulary(corpus:pd.Series):
    #Assumes corpus is tokenized with tokens being separated by whitespace
    assert isinstance(corpus,pd.Series), 'Text must be a pandas dataframe!'
    joined_text = ' '.join(corpus.values)
    vocabulary = {i:token for i,token in enumerate(set(joined_text.split()))}
    return vocabulary

def create_generator(corpus:pd.Series):
    vocabulary = create_vocabulary(corpus)
    for item in vocabulary.keys():
        filter = corpus.str.contains(vocabulary[item])
        yield process_text(corpus[filter])

def create_TFIDF_KNN(corpus:pd.Series,):
    context_generator = create_generator(corpus)
    vocab_contexts = [item for item in context_generator]
    pipeline_parts = [('vectorizer',TfidfVectorizer()),('knn',KNeighborsClassifier())]
    pipeline = Pipeline(pipeline_parts)
    return pipeline.fit(vocab_contexts)









