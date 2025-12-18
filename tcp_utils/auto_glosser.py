import pandas as pd
import numpy as np
import nltk, spacy
import Data_Processor as dp
from typing import Tuple,List
from nltk.lm.preprocessing import flatten
from nltk.lm.vocabulary import Vocabulary



def get_vocabularies(corpus:pd.DataFrame,tokenizer:dp.Tokenizer) -> Tuple[nltk.lm.Vocabulary,nltk.lm.Vocabulary]:
    nlp = spacy.load('en_core_web_sm')
    tag_filter = {'PRP','PRP$','PUNCT','ADP','DT'}
    source_language = corpus.iloc[:,0].apply(lambda x: tokenizer.tokenize(x).split(' '))
    target_language = corpus.iloc[:,1].apply(lambda x: [token.lemma_.lower() for token in nlp(x) if
                                                        token.tag_ not in tag_filter and not token.is_punct])
    source_vocab = Vocabulary(flatten(source_language.tolist()))
    target_vocab = Vocabulary(flatten(target_language.tolist()))
    return (source_vocab,target_vocab)


def create_gloss_table(vocabularies: Tuple[nltk.lm.Vocabulary,nltk.lm.Vocabulary]) -> pd.DataFrame:
    assert len(vocabularies) == 2, 'You can only create a gloss table with two languages!'
    table_index = list(vocabularies[0].counts.keys())
    table_columns = list(vocabularies[1].counts.keys())
    table_shape = np.zeros((len(table_index),len(table_columns)))
    return pd.DataFrame(data=table_shape,index=table_index,columns=table_columns)

def fill_gloss_table(corpus:pd.DataFrame,gloss_table:pd.DataFrame,tokenizer:dp.Tokenizer) -> pd.DataFrame:
    nlp = spacy.load('en_core_web_sm')
    corpus_copy = corpus.copy()
    tag_filter = {'PRP','PRP$','PUNCT','ADP','DT'}
    corpus_copy.iloc[:,0] = corpus_copy.iloc[:,0].apply(lambda x: tokenizer.tokenize(x))
    corpus_copy.iloc[:,1] = corpus_copy.iloc[:,1].apply(lambda x: [token.lemma_.lower() for token in nlp(x) if
                                                        token.tag_ not in tag_filter and not token.is_punct])
    for _,row in corpus_copy.iterrows():
        source_sent = set(row.iloc[0].split(' '))
        target_sent = set(row.iloc[1])

        gloss_table.loc[list(source_sent),list(target_sent)] = gloss_table.loc[list(source_sent),list(target_sent)] + 1

    return gloss_table

def create_tf_table(gloss_table:pd.DataFrame) -> pd.DataFrame:
    gloss_table_copy = gloss_table.copy()
    gloss_table_copy[gloss_table_copy > 0] = np.log10(gloss_table_copy[gloss_table_copy > 0])
    gloss_table_copy[gloss_table_copy > 0] = gloss_table_copy[gloss_table_copy > 0] + 1
    return gloss_table_copy

def create_idf_vector(gloss_table:pd.DataFrame) -> pd.Series:
    bool_table = gloss_table > 0
    idf_denominator_vector = bool_table.sum(axis=0)
    idf_numerator_list = [len(bool_table.index)] * len(bool_table.columns)
    idf_numerator_vector = pd.Series(idf_numerator_list, index=idf_denominator_vector.index)
    idf_vector = np.log10(idf_numerator_vector.div(idf_denominator_vector))
    return pd.Series(idf_vector)

def create_tfidf_table(gloss_table:pd.DataFrame) -> pd.DataFrame:
    tf_table = create_tf_table(gloss_table)
    idf_vector = create_idf_vector(gloss_table).to_numpy()
    tfidf_table = tf_table.mul(idf_vector)
    return tfidf_table


class Glosser: #Base class for glosser
    def __init__(self,corpus:pd.DataFrame,tokenizer:dp.Tokenizer) -> None:
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.vocabularies = None
        self.gloss_table = None
        self.set_vocabularies()
        self.set_gloss_table()

    def set_vocabularies(self):
        self.vocabularies = get_vocabularies(self.corpus,self.tokenizer)

    def set_gloss_table(self):
        assert self.vocabularies is not None, 'You must create vocabularies first!'
        gloss_table = create_gloss_table(self.vocabularies)
        self.gloss_table = fill_gloss_table(self.corpus,gloss_table,self.tokenizer)

    def align_sentence(self,sentence:pd.Series) -> List[Tuple[str,str]]:
        raise NotImplementedError


class Tfidf_Glosser(Glosser):
    def __init__(self,corpus:pd.DataFrame,tokenizer:dp.Tokenizer) -> None:
        super().__init__(corpus,tokenizer)
        self.tfidf_table = None
        self.set_tfidf_table()

    def set_tfidf_table(self):
        assert self.vocabularies is not None, 'You must create vocabularies first!'
        assert self.gloss_table is not None, 'You must create a gloss table first!'
        self.tfidf_table = create_tfidf_table(self.gloss_table)

    def align_sentence(self,sentence:pd.Series) -> List[Tuple[str,str]]:
        nlp = spacy.load('en_core_web_sm')
        tag_filter = {'PRP', 'PRP$', 'PUNCT', 'ADP', 'DT'}
        tokenized_source = self.tokenizer.tokenize(sentence.iloc[0]).split(' ')
        tokenized_target = [token.lemma_.lower() for token in nlp(sentence.iloc[1]) if token.tag_ not in tag_filter
                            and not token.is_punct]
        tfidf_slice = self.tfidf_table.loc[tokenized_source,tokenized_target]
        print(tfidf_slice.idxmax(axis=1))



class Entropy_Glosser(Glosser):
    def __init__(self,corpus:pd.DataFrame,tokenizer:dp.Tokenizer) -> None:
        super().__init__(corpus,tokenizer)
        self.entropy_table = None
        self.set_entropy_table()

    def set_entropy_table(self):
        assert self.vocabularies is not None, 'You must create vocabularies first!'
        assert self.gloss_table is not None, 'You must create a gloss table first!'
        smoothed_table = self.gloss_table + 1
        probability_table = smoothed_table.div(self.gloss_table.sum(axis=1),axis=0)
        entropy_table = np.log2(probability_table)
        self.entropy_table = entropy_table * -1


    def align_sentence(self,sentence:pd.Series) -> List[Tuple[str,str]]:
        nlp = spacy.load('en_core_web_sm')
        tag_filter = {'PRP', 'PRP$', 'PUNCT', 'ADP', 'DT'}
        tokenized_source = self.tokenizer.tokenize(sentence.iloc[0]).split(' ')
        tokenized_target = [token.lemma_.lower() for token in nlp(sentence.iloc[1]) if token.tag_ not in tag_filter
                            and not token.is_punct]
        entropy_slice = self.entropy_table.loc[tokenized_source, tokenized_target]
        print(entropy_slice.idxmin(axis=1))









