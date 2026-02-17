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
        indexes = Counter([token for token in X.iloc[0].split() if token in self.model.index])
        columns = Counter([token for token in X.iloc[1].split() if token in self.model.columns])
        output = self.predict_recursively(indexes, columns)
        print(output)
        return output

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
        concordance_list = [df_column_to_string(X[key]) for key in X.keys()]
        vectorizer = CountVectorizer()
        tfidf_matrix = vectorizer.fit_transform(concordance_list)
        frequency_table = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, index=list(X.keys()),
                                                            columns=vectorizer.get_feature_names_out())
        frequency_table = frequency_table.sparse.to_dense()
        frequency_table = frequency_table + 1
        probability_table = frequency_table.div(frequency_table.sum(axis=1), axis=0)
        entropy_table = np.log2(probability_table)
        self.model = entropy_table * -1
        self.model['Entropy'] = (self.model * probability_table).sum(axis=1)
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

        elif len(X_list) > 1:
            token_glosses = self.model.loc[x_list,y_list]
            token_glosses = self.model.sort_values(by=['Entropy'])
            token_glosses = token_glosses.drop('Entropy',axis=1)
            results.append((token_glosses.index[0], token_glosses.iloc[0].idxmin()))
            X[token_glosses.index[0]] -= 1
            y[token_glosses.iloc[0].idxmin()] -= 1
            return results + self.predict_recursively(X, y)



