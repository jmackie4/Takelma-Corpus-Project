import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator,TransformerMixin
from nltk.lm.preprocessing import flatten
from nltk.lm.vocabulary import Vocabulary
from typing import List

def process_text(text:pd.Series) -> str:
    #Assumes input is tokenized with tokens being separated by whitespace
    assert isinstance(text,pd.Series), 'Text must be a pandas series object!'
    return ' '.join(text.values)


def create_vocabulary(corpus:pd.Series):
    #Assumes corpus is tokenized with tokens being separated by whitespace
    assert isinstance(corpus,pd.Series), 'Text must be a pandas dataframe!'
    processed_corpus = process_text(corpus)
    vocabulary = {i:token for i,token in enumerate(sorted(set(processed_corpus.split())))}
    return vocabulary

def create_generator(corpus:pd.Series):
    vocabulary = create_vocabulary(corpus)
    for i in range(len(vocabulary)):
        filter = corpus.str.contains(vocabulary[i])
        yield process_text(corpus[filter])

def create_TFIDF_KNN(corpus:pd.Series,):
    context_generator = create_generator(corpus)
    vocab_contexts = [item for item in context_generator]
    pipeline_parts = [('vectorizer',TfidfVectorizer()),('knn',KNeighborsClassifier())]
    pipeline = Pipeline(pipeline_parts)
    return pipeline.fit(vocab_contexts)



class Vector_Semantics:
    def __init__(self,corpus:pd.Series):
        self.corpus = corpus
        self.model = create_TFIDF_KNN(self.corpus)
        self.idx_2_token = create_vocabulary(self.corpus)
        self.token_2_idx = {item:key for key,item in self.idx_2_token.items()}

    def size(self):
        return self.model.named_steps['knn'].n_samples_fit_

    def get_neighbor(self,X:list[str]):
        #Assumes that tokens in list have already been normalized
        token_idxs: List[int] = [self.token_2_idx[token] for token in X]

        #For this next part I need to run thru the first part of the TFIDF KNN pipeline object to just get the TFIDF vectors
        temp_generator = create_generator(self.corpus)
        temp_vocabulary = [item for item in temp_generator]
        temp_tfidf = TfidfVectorizer()

        #now I get the tfidf matrix of the corpus so I can pick out the necessary vectors
        tfidf_matrix = temp_tfidf.fit_transform(temp_vocabulary)
        assert isinstance(tfidf_matrix,np.ndarray)
        input_slice = tfidf_matrix[token_idxs]
        output = self.model.named_steps['knn'].kneighbors(input_slice)
        print(output)










