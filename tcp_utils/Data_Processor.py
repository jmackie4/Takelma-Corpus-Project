import pandas as pd
import numpy as np
import nltk,os,itertools
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from typing import Dict,List
from sklearn.pipeline import Pipeline

#The following just takes a root folder that holds the two directories for the parallel texts
#and it returns a dictionary that has the source language dir path and target language dir path
class Corpus_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: str):
        text_directories = self.get_subdirectories(X)
        output_dict = {}
        while True:
            print(text_directories)
            user_choice = input(
                'Please pick which directory is the source language! Type the name exactly as you see it!')
            if user_choice in text_directories:
                break
            else:
                print('Please try again!')

        output_dict['source_lang_path'] = os.path.join(X, user_choice)
        text_directories.remove(user_choice)
        output_dict['target_lang_path'] = os.path.join(X, text_directories[0])
        return output_dict

    def get_subdirectories(self,X:str):
        with os.scandir(X) as entries:
            text_directories = [entry.name for entry in entries if entry.is_dir()]
            return text_directories

class Corpus_Loader(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self,X:Dict[str,str]): #Input is just directory paths. Each path leads to folder of txt files
        parallel_texts = self.get_parallel_text(*[file for file in os.listdir(X['source_lang_path'])],
                                                source_lang_path=X['source_lang_path'],
                                                target_lang_path=X['target_lang_path'])
        for text in parallel_texts:
            yield text

    def get_parallel_text(self,*args,source_lang_path=None,target_lang_path=None):
    #args are individual file names, kwargs are source and target folders
       if source_lang_path and target_lang_path:
        for arg in args:
            with open(os.path.join(source_lang_path,arg),'r',encoding='utf-8') as f:
                source_sentences = [line.strip() for line in f]

            with open(os.path.join(target_lang_path,arg),'r',encoding='utf-8') as f:
                target_sentences = [line.strip() for line in f]
            yield arg,source_sentences,target_sentences
        else:
            print(f'Here\'s source:{source_lang_path}, and here\'s target:{target_lang_path}')

def create_corpus():
    while True:
        user_path = input('Please give the path of the root directory! ')
        if os.path.isdir(user_path):
            break
        else:
            print('The provided path is not a directory!')

    pipeline = Pipeline([('corpus_transformer', Corpus_Transformer()),
                         ('corpus_loader', Corpus_Loader())])
    pipeline_output = pipeline.fit_transform(user_path)
    source_sentences = []
    target_sentences = []
    tuples_for_multindex= []
    for text in pipeline_output:
        source_sentences.extend(text[1])
        target_sentences.extend(text[2])
        tuples_for_multindex.extend(list(zip(itertools.repeat(text[0]),[i for i,_ in enumerate(text[1])])))

    multiindex = pd.MultiIndex.from_tuples(tuples_for_multindex)
    corpus_dataframe = pd.DataFrame({'source_sentences': source_sentences,
                                     'target sentences': target_sentences},
                                    index=multiindex)
    return corpus_dataframe


class Tokenizer_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, pattern: str = None):
        self._pattern = pattern
        self._nlp = spacy.load('en_core_web_sm')
        self._tag_filter = {'PRP', 'PRP$', 'PUNCT', 'ADP', 'DT'}

    def fit(self, X, y=None):
        if self._pattern is not None:
            self.tokenizer_ = nltk.RegexpTokenizer(self._pattern)
        else:
            pass

        return self

    def transform(self, X: pd.DataFrame):
        X.iloc[:,0] = X.iloc[:,0].apply(lambda x: ' '.join(self.tokenizer_.tokenize(x.lower())))
        X.iloc[:,1] = X.iloc[:,1].apply(lambda x: ' '.join([token.lemma_.lower() for token in self._nlp(x)
                                                         if token.tag_ not in self._tag_filter and not
                                                         token.is_punct]))
        return X










