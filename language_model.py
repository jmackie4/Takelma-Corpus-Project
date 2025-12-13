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


def generate_text(func):
    def wrapper(model,n=5,sents:int = 10):
        i = 0
        text = []
        while i<sents:
            text.append(func(model,n=n))
            i += 1
        print(text)
        return text
    return wrapper


@generate_text
def generate_sentence(model:nltk.lm.api.LanguageModel,n:int=15) -> List[str]:
    assert model.vocab and model.counts, 'You need a trained model to generate a sentence!'
    context = ['<s>']*(model.order-1)
    i = 0
    while i < n and context[-1] != '</s>':
        context = context + [model.generate(num_words=1,text_seed=context)]
        i += 1
    print(context[model.order-1:-1])
    return context[model.order-1:-1]


#/Users/justinmackie/Dropbox/Mac/Desktop/Coding Projects/Takelma Corpus Project
#Parallel Texts
#\w+(?:'\w+)?(?:|[--‐]+)?\w+(?:'\w+)?









