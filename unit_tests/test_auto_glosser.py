import unittest
from tcp_utils import auto_glosser as ag
import pandas as pd
import numpy as np
from collections import Counter

class Test_Column_to_String(unittest.TestCase):
    def setUp(self):
        self.three_column_string_df = pd.DataFrame({'column1':['one','two','three'],
                                             'column2':['four','five','six'],
                                             'column3':['seven','eight','nine']})
        self.two_column_string_df = pd.DataFrame({'column1':['one','two','three'],
                                                  'column2':['four','five','six'],})
        self.three_column_string_int_df = pd.DataFrame({'column1':['one',2,'three'],
                                                        'column2':[4,'five','six'],
                                                        'column3':['seven','eight',9],})
    def test_valid_input(self):
        output = ag.df_column_to_string(self.three_column_string_df)
        self.assertIsInstance(output,str)

    def test_mixed_datatypes(self):
        output = ag.df_column_to_string(self.three_column_string_int_df)
        self.assertIsInstance(output,str)

    def test_specific_column(self):
        output1 = ag.df_column_to_string(self.two_column_string_df,column_idx=0)
        output2 = ' '.join(self.two_column_string_df.iloc[:,0].values.tolist())

        self.assertEqual(output1,output2)


class Test_Base_Glosser(unittest.TestCase):
    def setUp(self):
        self.three_column_string_df = pd.DataFrame({'column1':['one','two','three'],
                                                    'column2':['four','five','six'],
                                                    'column3':['seven','eight','nine']})
        self.test_series = pd.Series(['one two three','four five six'])
    def test_create_glosser(self):
        output = ag.BaseGlosser()
        self.assertIsInstance(output,ag.BaseGlosser)

    def test_base_fit(self):
        glosser = ag.BaseGlosser()
        glosser.fit(self.three_column_string_df)
        self.assertIsInstance(glosser.model,pd.DataFrame)

    def test_base_predict(self):
        glosser = ag.BaseGlosser()
        glosser.fit(self.three_column_string_df)
        output = glosser.predict(self.test_series)
        self.assertIsInstance(output,list)

    def test_valid_input(self):
        glosser = ag.BaseGlosser()
        glosser.fit(self.three_column_string_df)
        with self.assertRaises(AttributeError):
            output = glosser.predict(self.three_column_string_df)

    def test_base_predict_recursively_base_case(self):
        glosser = ag.BaseGlosser()
        glosser.fit(self.three_column_string_df)
        test_x = Counter([glosser.model.index.values[0]])
        test_y = Counter(['column1','column2'])
        output = glosser.predict_recursively(test_x,test_y)
        self.assertIsInstance(output,list)
        self.assertEqual(output,[(list(test_x.keys())[0],
                                  glosser.model.loc[list(test_x.keys())[0],list(test_y.keys())].idxmax())])

    def test_base_predict_recursively_recursive_case(self):
        glosser = ag.BaseGlosser()
        glosser.fit(self.three_column_string_df)
        test_x = Counter(glosser.model.index.values)
        test_y = Counter(['column1','column2','column3'])
        output = glosser.predict_recursively(test_x,test_y)
        self.assertTrue(all(isinstance(item,tuple) for item in output))
        self.assertTrue(all(len(item) == 2 for item in output))
        self.assertEqual(len(output),3)

    def test_base_predict_recursively_uneven_matches(self):
        glosser = ag.BaseGlosser()
        glosser.fit(self.three_column_string_df)
        test_x = Counter(glosser.model.index.values)
        test_x[3] += 1
        test_y = Counter(['column1', 'column2', 'column3'])
        output = glosser.predict_recursively(test_x, test_y)
        self.assertEqual(len(output), 4)


class Test_Create_Frequency_Table(unittest.TestCase):
    def setUp(self):
        self.valid_dictionary = {'test one': pd.DataFrame({'column1':['one','two','three'],
                                                                 'column2':['four','five','six'],
                                                                 'column3':['seven','eight','nine']})}
        self.test_dataframe = pd.DataFrame({'column1':['one','two','three'],
                                            ' column2':['four','five','six'],})
    def test_valid_input(self):
        output = ag.create_frequency_table(self.valid_dictionary)
        self.assertIsInstance(output,pd.DataFrame)

    def test_invalid_dataframe_input(self):
        with self.assertRaises(pd.errors.IndexingError):
            ag.create_frequency_table(self.test_dataframe)


class Test_Create_Probability_Table(unittest.TestCase):
    def setUp(self):
        self.test_dataframe = pd.DataFrame({1:[1,2,3],
                                           2:[4,5,6],
                                           3:[7,8,9]},
                                           )

    def test_valid_input(self):
        output = ag.create_probability_table(self.test_dataframe)
        self.assertTrue(output.sum().sum() == len(self.test_dataframe))

    def test_invalid_input(self):
        test_np_array = np.array([[1,2,3],[4,5,6]])
        with self.assertRaises(AttributeError):
            ag.create_probability_table(test_np_array)


class Test_Create_Entropy_Table(unittest.TestCase):
    def setUp(self):
        self.probability_table = pd.DataFrame({1:[0.9,0.7,0.5],
                                                    2:[0.05,0.15,0.25],
                                                    3:[0.05,0.15,0.25],
                                                    }
                                                   )

    def test_valid_input(self):
        output = ag.create_entropy_table(self.probability_table)
        reversed_entropy_table = output ** 2
        test_equation = (self.probability_table.sum().sum()) - (reversed_entropy_table.sum().sum())
        self.assertTrue(bool(test_equation < 0.01))

    def test_get_entropy_values(self):
        output = ag.create_entropy_table(self.probability_table)
        self.assertIsInstance(output['Entropy'],pd.Series)

