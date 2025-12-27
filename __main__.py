import pandas as pd
from tcp_utils import Data_Processor as dp
from tcp_utils import language_model as lm
from tcp_utils import auto_glosser as ag
from tcp_utils import vector_semantics as vs



class Hub():
    def __init__(self):
        self.processor = dp.DataProcessor()
        self.tokenizer = dp.Tokenizer()
        self.corpus = self.processor.get_corpus()
        self.language_model = self.create_model()
        self.aligner = self.create_aligner()
        self.vector_space = self.create_vector_space()


#Main Block of Hub Object
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

    def set_corpus(self,new_corpus:pd.DataFrame):
        assert isinstance(new_corpus,pd.DataFrame), 'Corpus must be a dataframe!'
        self.corpus = new_corpus

    def find_token_sequence(self):
        assert self.corpus is not None, 'Can\'t find a token sequence in an empty corpus!'
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


#N-Gram Block of the Hub Object
    def create_model(self):
        while True:
            user_num_grams = input('Please enter what kind of n-gram model you\'d like to make by giving a number: ')
            if user_num_grams == '':
                self.language_model = lm.create_model(self.corpus,self.tokenizer)
                break
            else:
                try:
                    int(user_num_grams)
                except ValueError:
                    print('Please enter a valid number!')
                return lm.create_model(self.corpus,self.tokenizer,n=int(user_num_grams))


    def get_model(self):
        return self.language_model

    def generate_text(self):
        return lm.generate_sentence(self.language_model)


#Aligner Block of the Hub Object
    def create_aligner(self):
        available_aligners = {i:glosser for i,glosser in enumerate({ag.Entropy_Glosser,ag.Tfidf_Glosser})}
        while True:
            for i,aligner in available_aligners.items():
                print(f'{i}: {aligner}')
            user_choice = input('Please enter your choice of aligner using the integer associated with the aligner: ')
            try:
                int(user_choice) in available_aligners
            except ValueError:
                print('Please enter a valid integer!')
            else:
                break
        return available_aligners[int(user_choice)](self.corpus,self.tokenizer)

    def get_aligner(self):
        return self.aligner

    def align_text(self):
        assert self.aligner is not None, 'You need to set an aligner first before you start aligning stuff!!!'
        user_text = self.processor.get_text()
        for _,row in user_text.iterrows():
            self.aligner.align_sentence(row)


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
        tokenized_corpus = self.corpus.iloc[:,int(user_choice)].apply(lambda x: self.tokenizer.tokenize(x))
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










        
        

    
        
        
