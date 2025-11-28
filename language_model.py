import pandas as pd
import numpy as np
import os, nltk, re, random
from collections import defaultdict, Counter
import Data_Processor as dp
import util_datastructs as ud
from typing import List


class Language_Model:
    def __init__(self, corpus: dp.DataProcessor, tokenizer: dp.Tokenizer):
        self.corpus = corpus.get_corpus()
        self.tokenizer = tokenizer

    def set_language(self):
        assert isinstance(self.corpus, pd.DataFrame)
        while True:
            print(f'{self.corpus.iloc[0, 0]} | {self.corpus.iloc[0, 1]}')
            user_choice = input('Please select the language with 0 or 1: ')
            if int(user_choice) in [0, 1]:
                break
            else:
                print('Please give a valid answer!')
        self.corpus = self.corpus.iloc[:, int(user_choice)]
        print('New Corpus \n {}'.format(self.corpus))

    def tokenize_corpus(self):
        assert isinstance(self.corpus, pd.Series)
        converted_corpus = self.corpus.tolist()
        tokenized_corpus = [self.tokenizer.tokenize(x) for x in converted_corpus]
        return tokenized_corpus

    def organize_corpus(self):
        pretrain_corpus: List[str] = []
        tokenized_corpus = self.tokenize_corpus()
        assert isinstance(tokenized_corpus, list)
        tokenized_corpus = [x.split(' ') for x in tokenized_corpus]
        for item in tokenized_corpus:
            item.insert(0, '<START>')
            item.append('<END>')
            pretrain_corpus.extend(item)
        return pd.Series(pretrain_corpus)


class N_Gram_Model(Language_Model):
    def __init__(self, corpus: dp.DataProcessor, tokenizer: dp.Tokenizer, n=2):
        super().__init__(corpus, tokenizer)
        self.n = n
        self.set_language()
        self.n_gram_table = None

    def create_n_gram_table(self):
        assert isinstance(self.corpus, pd.Series)
        organized_corpus: pd.Series = self.organize_corpus()
        n_gram_counts: defaultdict[str, Counter] = defaultdict(Counter)
        n_gram_index: List[str] = list(organized_corpus.unique())
        i = 0
        while True:
            try:
                organized_corpus.iloc[i + self.n]
            except IndexError:
                break
            else:
                if self.n == 1:
                    n_gram_counts[organized_corpus.iloc[i]] += 1
                    i += 1
                elif self.n > 1:
                    n_gram_slice = organized_corpus.iloc[i:i + self.n].tolist()
                    assert len(n_gram_slice) == self.n
                    n_gram_counts[' '.join(n_gram_slice[:-1])][n_gram_slice[-1]] += 1
                    i += 1

        if self.n == 1:
            final_output: pd.Series = pd.Series(n_gram_counts, index=n_gram_index)
            final_output = final_output.div(final_output.sum())
            self.n_gram_table = final_output
            print(self.n_gram_table)
        else:
            final_output: pd.DataFrame = pd.DataFrame.from_dict(n_gram_counts, orient='index', columns=n_gram_index)
            filled_output = final_output.fillna(0)
            filled_output = filled_output + 1
            self.n_gram_table = filled_output.div(filled_output.sum(axis=1), axis=0)
            print(self.n_gram_table)

    def predict(self):
        assert self.n_gram_table is not None
        sent = []

    def create_and_predict(self):
        self.create_n_gram_table()
        self.predict()