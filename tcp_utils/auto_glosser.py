import pandas as pd
import numpy as np
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from collections import Counter

def df_slice_to_string(dataframe:pd.DataFrame) -> str:
    return ' '.join(dataframe.iloc[:,1].values.tolist())

class tfidf_glosser(BaseEstimator):
    def __init__(self):
        pass

    def fit(self,X:Dict[str,pd.DataFrame],y=None):
        concordance_list = [df_slice_to_string(X[key]) for key in X.keys()]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(concordance_list)
        self.model_ = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix,index=list(X.keys()), columns=vectorizer.get_feature_names_out())
        return self


    def predict(self,X:pd.Series):
        indexes = list(set(X.iloc[0].split()))
        columns = [token for token in X.iloc[1].split() if token in self.model_.columns]
        model_slice = self.model_.loc[indexes,columns]
        model_slice = model_slice.sparse.to_dense()
        return model_slice.idxmax(axis=1)


class entropy_glosser(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X: Dict[str, pd.DataFrame], y=None):
        concordance_list = [df_slice_to_string(X[key]) for key in X.keys()]
        vectorizer = CountVectorizer()
        tfidf_matrix = vectorizer.fit_transform(concordance_list)
        frequency_table = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, index=list(X.keys()),
                                                            columns=vectorizer.get_feature_names_out())
        frequency_table = frequency_table.sparse.to_dense()
        frequency_table = frequency_table + 1
        probability_table = frequency_table.div(frequency_table.sum(axis=1), axis=0)
        entropy_table = np.log2(probability_table)
        self.model_ = entropy_table * -1
        self.model_['Entropy'] = (self.model_ * probability_table).sum(axis=1)
        print(self.model_.head())
        return self

    def predict(self, X: pd.Series):
        indexes = Counter(X.iloc[0].split())
        columns = Counter([token for token in X.iloc[1].split() if token in self.model_.columns])
        self.predict_recursively(indexes, columns)

    def predict_recursively(self, X: Counter, y: Counter):
        X_list = [key for key,value in X.items() if value > 0]
        y_list = [key for key,value in y.items() if value > 0]
        if len(X_list) == 0 or len(y_list) == 0:
            print('Done with this sentence!!!')

        elif len(X_list) == 1:
            token_glosses = self.model_.loc[X_list[0],y_list]
            print(f'{X_list[0]} {token_glosses.idxmin()}')

        elif len(X_list) > 1:
            token_glosses = self.model_.sort_values(by=['Entropy'])
            token_glosses = token_glosses.drop('Entropy',axis=1)
            token_glosses = token_glosses.loc[X_list,y_list]
            print(f'{token_glosses.index[0]} {token_glosses.iloc[0].idxmin()}')
            X[token_glosses.index[0]] -= 1
            y[token_glosses.iloc[0].idxmin()] -= 1
            self.predict_recursively(X, y)



