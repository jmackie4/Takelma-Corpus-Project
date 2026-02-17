import pandas as pd
import numpy as np
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from collections import Counter

def df_column_to_string(dataframe:pd.DataFrame,column_idx=1) -> str:
    return ' '.join(dataframe.iloc[:,column_idx].astype(str).values.tolist())


class BaseGlosser(BaseEstimator):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        self.model = pd.DataFrame(X)
        return self

    def predict(self,X,y=None):
        indexes = Counter(set([token for token in X.iloc[0].split() if token in self.model.index]))
        columns = Counter(set([token for token in X.iloc[1].split() if token in self.model.columns]))
        return self.predict_recursively(indexes, columns)

    def predict_recursively(self,X:Counter,y:Counter):
        x_list = [key for key,value in X.items() if value > 0]
        y_list = [key for key,value in y.items() if value > 0]
        results = []
        if x_list == [] or y_list == []:
            if y_list == []:
                return results + [(item,'no gloss') for item in x_list if x_list != []]
            elif x_list == []:
                return results + [('no source',item) for item in y_list if y_list != []]


        elif len(x_list) == 1:
            return results + [(x_list[0],self.model.loc[x_list[0],y_list].idxmax())]

        elif len(x_list) > 1:
            results.append((x_list[0],self.model.loc[x_list[0],y_list].idxmax()))
            X[x_list[0]] -= 1
            y[self.model.loc[x_list[0],y_list].idxmax()] -= 1
            return results + self.predict_recursively(X,y)


class tfidf_glosser(BaseGlosser):
    def __init__(self):
        pass

    def fit(self,X:Dict[str,pd.DataFrame],y=None):
        concordance_list = [df_column_to_string(X[key]) for key in X.keys()]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(concordance_list)
        self.model = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix,index=list(X.keys()), columns=vectorizer.get_feature_names_out())
        return self


class entropy_glosser(BaseGlosser):
    def __init__(self):
        pass

    def fit(self, X: Dict[str, pd.DataFrame], y=None):
        self.model = transform_dict_to_entropy_table(X)
        print(self.model.head())
        return self

    def predict_recursively(self, X: Counter, y: Counter):
        x_list = [key for key, value in X.items() if value > 0]
        y_list = [key for key, value in y.items() if value > 0]
        results = []
        if x_list == [] or y_list == []:
            if y_list == []:
                return results + [(item, 'no gloss') for item in x_list if x_list != []]
            elif x_list == []:
                return results + [('no source', item) for item in y_list if y_list != []]

        elif len(x_list) == 1:
            return results + [(x_list[0], self.model.loc[x_list[0], y_list].idxmin())]

        elif len(x_list) > 1:
            y_list_with_entropy = y_list+['Entropy']
            token_glosses = self.model.loc[x_list,y_list_with_entropy]
            token_glosses = token_glosses.sort_values(by=['Entropy'])
            token_glosses = token_glosses.drop('Entropy',axis=1)
            results.append((token_glosses.index[0], token_glosses.iloc[0].idxmin()))
            X[token_glosses.index[0]] -= 1
            y[token_glosses.iloc[0].idxmin()] -= 1
            return results + self.predict_recursively(X, y)


def transform_dict_to_entropy_table(X:Dict[str,pd.DataFrame]):
    frequency_table = create_frequency_table(X)
    probability_table = create_probability_table(frequency_table)
    return create_entropy_table(probability_table)


def create_frequency_table(X:Dict[str,pd.DataFrame]):
    input_for_vectorizer = [df_column_to_string(X[key]) for key in X.keys()]
    vectorizer = CountVectorizer()
    sparse_matrix = vectorizer.fit_transform(input_for_vectorizer)
    dense_array_matrix = sparse_matrix.toarray()
    output = pd.DataFrame(dense_array_matrix,index=list(X.keys()),columns=vectorizer.get_feature_names_out())
    output = output + 1
    return output


def create_probability_table(freq_table:pd.DataFrame):
    return freq_table.div(freq_table.sum(axis=1),axis=0)


def create_entropy_table(probability_table:pd.DataFrame):
    entropy_table =  np.log2(probability_table) * -1
    entropy_vector = (entropy_table * probability_table).sum(axis=1)
    entropy_table['Entropy'] = entropy_vector
    return entropy_table

