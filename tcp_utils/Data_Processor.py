import pandas as pd
import nltk,os
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from typing import Dict
from sklearn.pipeline import Pipeline

#The following just takes a root folder that holds the two directories for the parallel texts
#and it returns a dictionary that has the source language dir path and target language dir path
class Corpus_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: str):
        with os.scandir(X) as entries:
            text_directories = [entry.name for entry in entries if entry.is_dir()]

        output_dict = {}
        while True:
            print(text_directories)
            user_choice = input(
                'Please pick which directory is the source language! Type the name exactly as you see it!')
            if user_choice in text_directories:
                break
            else:
                print('Please try again!')
        output_dict['source_language'] = os.path.join(X, user_choice)
        text_directories.remove(user_choice)
        output_dict['target_language'] = os.path.join(X, text_directories[0])
        return output_dict



class Corpus_Loader(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self

    def transform(self,X:Dict[str,str]):
        ''' This corpus loader transformer needs to be with the Corpus transformer object in a
        pipeline in order to work!'''
        i = 0
        source_sents = []
        target_sents = []
        text_idxs = {}
        for file in os.listdir(X['source_language']):
            with open(os.path.join(X['source_language'],file),'r',encoding='utf-8') as f:
                  source_sents.extend([line[:-1] for line in f.readlines()])
                  text_idxs[file] = i,len(source_sents)
                  i = len(source_sents)

            with open(os.path.join(X['target_language'],file),'r',encoding='utf-8') as f:
                  target_sents.extend([line.rstrip('\n') for line in f.readlines()])

        output_dict = {'text_idxs':text_idxs,'source_sents':source_sents,'target_sents':target_sents}
        return output_dict



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
    corpus_dataframe = pd.DataFrame({'source_language': pipeline_output['source_sents'],
                                     'target_language': pipeline_output['target_sents'],
                                     })
    return corpus_dataframe,pipeline_output['text_idxs']



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
        X.iloc[:,0] = X.iloc[:, 0].apply(lambda x: ' '.join(self.tokenizer_.tokenize(x.lower())))
        X.iloc[:,1] = X.iloc[:, 1].apply(lambda x: ' '.join([token.lemma_.lower() for token in self._nlp(x)
                                                         if token.tag_ not in self._tag_filter and not
                                                         token.is_punct]))
        return X










