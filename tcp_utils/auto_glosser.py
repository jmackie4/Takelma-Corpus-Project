import pandas as pd
import numpy as np
import nltk, spacy
from . import Data_Processor as dp
from typing import Tuple,List,Dict
from nltk.lm.preprocessing import flatten
from nltk.lm.vocabulary import Vocabulary
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

class tf_idf_maker(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.model = TfidfVectorizer()

    def fit(self,X,y=None):
        return self

    def transform(self,X:Dict[str,pd.DataFrame]):
        corpus = []
        token_2_idx = {token:idx for idx,token in enumerate(X.keys())}
        for token in X.keys():
            corpus.append([set(row.iloc[:,1].split()) for row in X[token].iterrows()])
        return self.model.fit_transform(corpus)

class tfidf_glosser(BaseEstimator):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        self.model = X
        return self

    def predict(self,X:pd.Series):
        model_slice = self.model.loc[list(set(X.iloc[:,0].split())),list(set(X.iloc[:,1].split()))]
        return model_slice.idxmax(axis=1)


class entropy_glosser(BaseEstimator):
    def __init__(self):
        self.model = CountVectorizer()

    def fit(self,X):
        corpus = []
        token_2_idx = {token: idx for idx, token in enumerate(X.keys())}
        for token in X.keys():
            corpus.append([set(row.iloc[:, 1].split()) for row in X[token].iterrows()])
        count_table = self.model.fit_transform(corpus)
        count_table = count_table + 1
        count_table.div(count_table.sum(axis=1), axis=0)
        entropy_table = np.log(count_table)
        self.entropy_table_ = entropy_table * -1
        return self

    def predict(self,X:pd.Series):
        model_slice = self.entropy_table_.loc[list(set(X.iloc[:, 0].split())), list(set(X.iloc[:, 1].split()))]
        return model_slice.idxmax(axis=1)







