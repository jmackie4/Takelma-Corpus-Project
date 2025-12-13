import pandas as pd
import numpy as np
import os, nltk, re, random
from collections import defaultdict, Counter
import Data_Processor as dp
import util_datastructs as ud
from typing import List
from nltk.util import ngrams,everygrams
from nltk.lm import MLE,StupidBackoff
from nltk.lm.preprocessing import pad_sequence,pad_both_ends,flatten,padded_everygram_pipeline

def create_model(corpus:pd.DataFrame,tokenizer:dp.Tokenizer,n:int=2):
    while True:
        print(f'{corpus.iloc[0, 0]} | {corpus.iloc[0, 1]}')
        user_choice = input('Please choose one of the following options: ')
        try:
            int(user_choice) in [0,1]
        except ValueError:
            print('Please pick a valid option!')
        else:
            break
    user_language = corpus.iloc[:,int(user_choice)]
    tokenized_corpus = user_language.apply(lambda x: tokenizer.tokenize(x).split(' '))
    n_grams,vocabulary = padded_everygram_pipeline(n,tokenized_corpus.tolist())
    model = StupidBackoff(order=n)
    model.fit(n_grams,vocabulary)
    return model

def generate_sentence(model:nltk.lm.api.LanguageModel,starting_seed=None,n:int=10) -> List[str]:
    assert model.vocab and model.counts, 'You need a trained model to generate a sentence!'
    if starting_seed:
        output = model.generate(num_words=n,text_seed=[starting_seed])
    else:
        output = model.generate(num_words=n,text_seed=['<s>'])

    print(output)
    return output













processor = dp.DataProcessor()
tokenizer = dp.Tokenizer()
corpus = processor.get_corpus()
model = create_model(corpus,tokenizer,n=2)
generate_sentence(model,n=15)

#/Users/justinmackie/Dropbox/Mac/Desktop/Coding Projects/Takelma Corpus Project
#Parallel Texts
#\w+(?:'\w+)?(?:|[--‐]+)?\w+(?:'\w+)?









