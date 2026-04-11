import pandas as pd
import re
from tcp_utils import Data_Processor as dp
from tcp_utils import language_model as lm
from tcp_utils import auto_glosser as ag
from tcp_utils import vector_semantics as vs
from nltk.lm.vocabulary import Vocabulary




class Hub():
    def __init__(self):
        self.corpus = dp.create_corpus()
        self.create_tokenized_corpus()
        self.create_vocabularies()
        self.create_concordances()
        self.create_model()
        self.create_aligner()
        self.vector_space = self.create_vector_space()


#Constructor Setup Block of Hub Object
    def create_tokenized_corpus(self):
        assert self.corpus is not None, 'Cannot create the necessary tokenized corpus since there\'s no corpus to tokenize!'
        while True:
            user_pattern = input('Please enter your regular expression pattern for tokenization:\n')
            if user_pattern == '':
                print('Please enter a regular expression pattern and not an empty string!')
            else:
                break
        tokenizer = dp.Tokenizer_Transformer(pattern=user_pattern)
        self._tokenized_corpus = tokenizer.fit_transform(self.corpus)

    def create_vocabularies(self):
        source_vocab_obj = Vocabulary(' '.join(self._tokenized_corpus.iloc[:,0].values).split())
        target_vocab_obj = Vocabulary(' '.join(self._tokenized_corpus.iloc[:,1].values).split())
        self.source_vocab,self.target_vocab = list(source_vocab_obj.counts.keys()),list(target_vocab_obj.counts.keys())

    def create_concordances(self):
        source_concordances = {}
        target_concordances = {}
        for token in self.source_vocab:
            # Escape special regex characters in the token
            source_concordances[token] = self._tokenized_corpus[self._tokenized_corpus.iloc[:,0].str.contains(re.escape(token), regex=True)]
        for token in self.target_vocab:
            # Escape special regex characters in the token
            target_concordances[token] = self._tokenized_corpus[self._tokenized_corpus.iloc[:,1].str.contains(re.escape(token), regex=True)]
        self.source_concordances,self.target_concordances = source_concordances,target_concordances

#Main Block of Hub Object
    def get_text(self):
        idx_2_titles = {i:title for i,title in enumerate(self.corpus.index.unique(level=0))}
        for i,title in idx_2_titles.items():
            print(f'{i}: {title}')
        while True:
            user_choice = input('Please select which text you want by inputting the number! ')
            try:
                output = idx_2_titles[int(user_choice)]
                break
            except ValueError:
                print('Please enter a valid number!')
            except KeyError:
                print('Please enter a valid number!')
        for _,row in self.corpus.loc[idx_2_titles[int(user_choice)]].iterrows():
            print(f'{row.iloc[0]} || {row.iloc[1]}')


    def get_titles(self) -> None:
        for _,title in enumerate(self.corpus.index.unique(level=0)):
            print(title[:-4])


    def set_corpus(self,new_corpus:pd.DataFrame):
        assert isinstance(new_corpus,pd.DataFrame), 'Corpus must be a dataframe!'
        self.corpus = new_corpus
        self.create_tokenized_corpus()
        self.create_vocabularies()
        self.create_concordances()



    def find_token_sequence(self):
        assert self.corpus is not None, 'Can\'t find a token sequence in an empty corpus!'
        while True:
            print(f'{self.corpus.iloc[0, 0]} | {self.corpus.iloc[0, 1]}')
            user_choice = input('Please select the language with 0 or 1: ')
            if int(user_choice) in [0, 1]:
                break
            else:
                print('Please give a valid answer!')

        sents = self._tokenized_corpus.iloc[:, int(user_choice)]
        user_sequence = input('Please enter your sequence: ')
        filter = sents.str.contains(re.escape(user_sequence.lower()), regex=True, case=False)
        for _,row in self.corpus[filter].iterrows():
            print(f"{row.iloc[0]} | {row.iloc[1]}")



#N-Gram Block of the Hub Object
    def create_model(self):
        while True:
            user_num_grams = input('Please enter what kind of n-gram model you\'d like to make by giving a number: ')
            if user_num_grams == '':
                self.language_model = lm.create_model(self._tokenized_corpus)
                break
            else:
                try:
                    int(user_num_grams)
                except ValueError:
                    print('Please enter a valid number!')
                else:
                    self.language_model = lm.create_model(self._tokenized_corpus)
                    break


    def get_model(self):
        return self.language_model

    def generate_text(self):
        return lm.generate_sentence(self.language_model)


#Aligner Block of the Hub Object
    def create_aligner(self):
        available_aligners = {i:glosser for i,glosser in enumerate(['tfidf_glosser','entropy_glosser'])}
        while True:
            for i,aligner in available_aligners.items():
                print(f'{i}: {aligner}')
            user_choice = input('Please enter your choice of aligner using the integer associated with the aligner: ')
            if int(user_choice) in available_aligners:
                break
            else:
                print('Please enter a valid integer!')

        if int(user_choice) == 0:
            aligner = ag.tfidf_glosser()
            aligner.fit(self.source_concordances)
            self.aligner = aligner

        elif int(user_choice) == 1:
            aligner = ag.entropy_glosser()
            aligner.fit(self.source_concordances)
            self.aligner = aligner


    def get_aligner(self):
        return self.aligner

    def align_text(self):
        assert self.aligner is not None, 'You need to set an aligner first before you start aligning stuff!!!'
        idx_2_titles = {i: title for i, title in enumerate(self.corpus.index.unique(level=0))}
        for i, title in idx_2_titles.items():
            print(f'{i}: {title}')
        while True:
            user_choice = input('Please select which text you want by inputting the number! ')
            try:
                output = idx_2_titles[int(user_choice)]
                break
            except ValueError:
                print('Please enter a valid number!')
            except KeyError:
                print('Please enter a valid number!')
        for _, row in self.corpus.loc[idx_2_titles[int(user_choice)]].iterrows():
            print(self.aligner.predict(row))


#Vector Semantic Block of Hub Object
    def create_vector_space(self):
        assert self.corpus is not None, 'Can\'t find a vector space in an empty corpus!'
        while True:
            print(f'{self.corpus.iloc[0, 0]} | {self.corpus.iloc[0, 1]}')
            user_choice = input('Please enter your choice of language to create a vector space for: ')
            if int(user_choice) in [0, 1]:
                break
            else:
                print('Please enter a valid choice!')
        tokenized_corpus = self._tokenized_corpus.iloc[:,int(user_choice)]
        return vs.Vector_Semantics(tokenized_corpus)

    def get_neighbors(self):
        assert self.vector_space is not None, 'Can\'t find neighbors in an empty vector space!!'
        user_words = input('Please enter the words you want to query here, make sure they\'re separated by whitespace:\n')
        X = [word for word in user_words.lower().split() if word in self.vector_space.token_2_idx]
        print(f'Here are the invalid words: {[word for word in user_words.split() if word not in X]}')

        print('Processing valid words...')
        print('Here are the neighbors!')
        self.vector_space.get_neighbor(X)





if __name__ == '__main__':
    main_hub = Hub()
    options = {'get text': main_hub.get_text,'get titles': main_hub.get_titles,
               'use n-gram model':main_hub.generate_text,
               'find sequence': main_hub.find_token_sequence,
               'use aligner': main_hub.align_text,
               'find neighbors': main_hub.get_neighbors,
               }
    while True:
        for i,item in enumerate(options):
            print(f'{i}: {item}',end='\n')
        users_choice = input('Please enter what you want to do: ')
        if users_choice.lower() in options :
            options[users_choice.lower()]()
        elif users_choice.lower() == 'exit':
            break
        else:
            print('Please enter a valid choice!')










        
        

    
        
        
