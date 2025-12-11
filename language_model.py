import pandas as pd
import numpy as np
import os, nltk, re, random
from collections import defaultdict, Counter
import Data_Processor as dp
import util_datastructs as ud
from typing import List

def organize_corpus(corpus:pd.DataFrame,tokenizer:dp.Tokenizer):
    while True:
        user_choice = input(f'{corpus.iloc[0,0]} | {corpus.iloc[0,1]}' + '\n' + 'Enter a choice: ')
        try:
            int(user_choice) in [0,1] == True
        except ValueError:
            print('Please enter a valid integer')
        else:
            break

    sentences = corpus.iloc[:,int(user_choice)].apply(lambda x: tokenizer.tokenize(x))

    formatted_corpus = []
    for sentence in sentences.tolist():
        tokens:List[str] = sentence.split(' ')
        tokens.insert(0,'<START>')
        tokens.append('<END>')
        formatted_corpus.extend(tokens)

    return formatted_corpus


def ngram_decorator(func):
    def wrapper(corpus:pd.DataFrame,tokenizer:dp.Tokenizer,n:int=2):
        models = {}
        formatted_corpus = organize_corpus(corpus,tokenizer)
        while n >= 1:
            models[f'{n}-gram model'] = func(formatted_corpus,n)
            print(f'{n}-gram model')
            print(models[f'{n}-gram model'])
            n -= 1
        return models
    return wrapper

@ngram_decorator
def create_ngrams(corpus:List[str],n):
    if n == 1:
        unigram_table = pd.Series(Counter(corpus))
        unigram_table.drop(['<START>','<END>'],inplace=True)
        return unigram_table.div(unigram_table.sum())

    else:
        n_grams = defaultdict(lambda: defaultdict(int))
        i = 0
        while i + n <= len(corpus):
            ngram_window = corpus[i:i+n]
            n_grams[' '.join(ngram_window[:-1])][ngram_window[-1]] += 1
            i += 1

        ngram_table = pd.DataFrame.from_dict(n_grams)
        ngram_table = ngram_table.fillna(0)
        ngram_table = ngram_table + 1
        return ngram_table.div(ngram_table.sum(axis=0),axis=1)


def generate(model,min_tokens:int = 3,max_tokens:int = 10) -> str:
    if isinstance(model,pd.Series):
        num_tokens = random.randint(min_tokens,max_tokens)
        sentence_stack = []
        for _ in range(num_tokens):
            sentence_stack.append(random.choices(model.index,weights=model.values,k=1)[0])
        return ' '.join(sentence_stack)

    else:
        sentence = ud.LinkedList()
        sentence.add_node('<START>')







processor = dp.DataProcessor()
corpus = processor.get_corpus()
tokenizer = dp.Tokenizer()
create_ngrams(corpus,tokenizer,3)

#/Users/justinmackie/Dropbox/Mac/Desktop/Coding Projects/Takelma Corpus Project
#Parallel Texts
#\w+(?:'\w+)?(?:|[--‐]+)?\w+(?:'\w+)?









