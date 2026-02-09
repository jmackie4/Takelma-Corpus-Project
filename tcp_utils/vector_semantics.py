import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

def convert_series_to_str(text:pd.Series) -> str:
    #Assumes input is tokenized with tokens being separated by whitespace
    return ' '.join(text.values.astype(str))

def convert_text_to_list(text:str) -> List[str]:
    return text.split(' ')

def create_ordered_dict_from_list(input:List[str]):
    assert isinstance(input,list),'The input for this function must be a list!'
    return {i:item for i,item in enumerate(sorted(set(input)))}

def create_ordered_dict_from_series(text:pd.Series):
    str_version = convert_series_to_str(text)
    list_version = convert_text_to_list(str_version)
    return create_ordered_dict_from_list(list_version)

def create_generator_from_series(corpus:pd.Series):
    str_version = convert_series_to_str(corpus)
    list_version = convert_text_to_list(str_version)
    vocabulary = create_ordered_dict_from_list(list_version)
    for _,item in vocabulary.items():
        series_filter = corpus.str.contains(item)
        yield convert_series_to_str(corpus[series_filter])

def create_TFIDF_KNN(corpus:pd.Series,):
    context_generator = create_generator_from_series(corpus)
    vocab_contexts = [item for item in context_generator]
    pipeline_parts = [('vectorizer',TfidfVectorizer()),('knn',NearestNeighbors())]
    pipeline = Pipeline(pipeline_parts)
    return pipeline.fit(vocab_contexts)



class Vector_Semantics:
    def __init__(self,corpus:pd.Series):
        self.corpus = corpus
        self.model = create_TFIDF_KNN(self.corpus)
        self.idx_2_token = create_ordered_dict_from_series(self.corpus)
        self.token_2_idx = {item:key for key,item in self.idx_2_token.items()}

    def size(self):
        return self.model.named_steps['knn'].n_samples_fit_

    def get_neighbor(self,X:list[str]):
        #Assumes that tokens in list have already been normalized
        token_idxs: List[int] = [self.token_2_idx[token] for token in X]

        #For this next part I need to run thru the first part of the TFIDF KNN pipeline object to just get the TFIDF vectors
        temp_generator = create_generator_from_series(self.corpus)
        temp_vocabulary = [item for item in temp_generator]
        temp_tfidf = TfidfVectorizer()

        #now I get the tfidf matrix of the corpus so I can pick out the necessary vectors
        tfidf_matrix = temp_tfidf.fit_transform(temp_vocabulary).toarray()
        assert isinstance(tfidf_matrix,np.ndarray)
        input_slice = tfidf_matrix[token_idxs]
        output = self.model.named_steps['knn'].kneighbors(input_slice)
        print(output)










