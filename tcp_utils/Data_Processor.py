import pandas as pd
import numpy as np
import nltk,os,re
from collections import defaultdict,Counter
from . import util_datastructs

def create_corpus():
    corpus_location = input('Please enter the main path that holds the corpus: ')
    txt_folder_1,txt_folder_2 = input(f'Please enter the names of the folders that hold the texts: {os.listdir(corpus_location)}' + '\n').split(',')
    i = 0
    texts = {}
    title_idxs = {}
    for file in os.listdir(os.path.join(corpus_location,txt_folder_1)):
        with open(os.path.join(corpus_location,txt_folder_1,file),'r',encoding='utf-8') as f:
            lang_1 = [line[:-1] for line in f.readlines()]

        with open(os.path.join(corpus_location,txt_folder_2,file),'r',encoding='utf-8') as f:
            lang_2 = [line[:-1] for line in f.readlines()]

        assert len(lang_1) == len(lang_2)
        texts[file[:-4]] = pd.DataFrame({'lang 1':lang_1,'lang 2':lang_2})
        title_idxs[file[:-4]] = (i,i+len(lang_1))
        i += len(lang_1)

    final_dataframe = pd.concat(texts.values())
    return final_dataframe,title_idxs

class DataProcessor():
    def __init__(self):
        self.corpus,self.title_idxs = create_corpus()

    def get_titles(self):
        return list(self.title_idxs.keys())

    def get_corpus(self):
        return self.corpus

    def get_text(self):
        while True:
            option_list = {i:key for i,key in enumerate(list(self.title_idxs.keys()))}
            for item in option_list.items():
                print(f'{item[0]}: {item[1]}',end='\n')
            user_choice = input('Please enter your choice by giving a number: ')
            if int(user_choice) in option_list:
                break
            else:
                print('Please enter a valid choice!')
        user_choice = self.title_idxs[option_list[int(user_choice)]]
        text = self.corpus.iloc[user_choice[0]:user_choice[1]]
        print(text)
        return text

class Tokenizer():
    def __init__(self):
        self.tokenizer = None
        self.pattern = None
        self.set_tokenizer()

    def set_tokenizer(self):
        user_pattern = input('Please enter your regex pattern: ')
        self.pattern = re.compile(r'''{}'''.format(user_pattern))
        self.tokenizer = nltk.RegexpTokenizer(user_pattern)

    def tokenize(self,text):
        return ' '.join(self.tokenizer.tokenize(text.lower()))














