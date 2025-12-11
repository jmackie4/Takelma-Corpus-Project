import pandas as pd
import numpy as np
import os,nltk,re,random
from collections import defaultdict,Counter
import Data_Processor as dp
import util_datastructs as ud
import language_model as lm
from typing import List



class Hub():
    def __init__(self):
        self.processor = dp.DataProcessor()
        self.tokenizer = dp.Tokenizer()
        self.corpus = None
        self.set_corpus()

    def get_processor(self):
        return self.processor

    def get_tokenizer(self):
        return self.tokenizer

    def get_text(self):
        self.processor.get_text()

    def get_titles(self):
        self.processor.get_titles()

    def get_corpus(self):
        assert self.corpus is not None
        return self.corpus

    def set_corpus(self):
        self.corpus = self.processor.get_corpus()

    def find_token_sequence(self):
        if self.corpus is None:
            self.set_corpus()
        while True:
            print(f'{self.corpus.iloc[0, 0]} | {self.corpus.iloc[0, 1]}')
            user_choice = input('Please select the language with 0 or 1: ')
            if int(user_choice) in [0, 1]:
                break
            else:
                print('Please give a valid answer!')

        sents = self.corpus.iloc[:, int(user_choice)]
        sents.apply(lambda x: self.tokenizer.tokenize(x))
        user_sequence = input('Please enter your sequence: ')
        filter = sents.str.contains(user_sequence.lower(), case=False)
        print(self.corpus[filter])








if __name__ == '__main__':
    main_hub = Hub()
    options = {'get text': main_hub.get_text,'get titles': main_hub.get_titles,'n_grams':lm.create_ngrams,
               'find sequence': main_hub.find_token_sequence}
    while True:
        for i,item in enumerate(options):
            print(f'{i}: {item}',end='\n')
        users_choice = input('Please enter your choice: ')
        if users_choice.lower() in options :
            options[users_choice.lower()]()
        elif users_choice.lower() == 'exit':
            break
        else:
            print('Please enter a valid choice!')










        
        

    
        
        
