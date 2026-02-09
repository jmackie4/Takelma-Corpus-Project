import unittest,types
from tcp_utils import auto_glosser as ag
import pandas as pd

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
        self.three_column_string_df_gold = 'four five six'
        self.two_column_string_df_gold = 'four five six'
        self.test_spaced_string = 'one two three four'
        self.test_num_list = ['one','two','three','four','five','six']

    def test_column_to_string(self):
        self.assertTrue(isinstance(ag.df_slice_to_string(self.three_column_string_df),str))
        self.assertEqual(ag.df_slice_to_string(self.three_column_string_df),self.three_column_string_df_gold)
        self.assertNotEqual(ag.df_slice_to_string(self.two_column_string_df),ag.df_slice_to_string(self.two_column_string_df.T))
        with self.assertRaises(AttributeError):
            ag.df_slice_to_string(self.test_spaced_string)
        with self.assertRaises(AttributeError):
            ag.df_slice_to_string(self.test_num_list)
        with self.assertRaises(TypeError):
            ag.df_slice_to_string(self.three_column_string_int_df)


class Test_Convert_Dict_To_List(unittest.TestCase):
    def setUp(self):
        self.test_dict = {'key1':pd.DataFrame({'column1':['one','two','three'],
                                             'column2':['four','five','six'],
                                             'column3':['seven','eight','nine']}),
                          'key2':pd.DataFrame({'column1':['apple','pineapple','coconut'],
                                               'column2':['spinach','potato','cauliflower'],
                                               }),
                          'key3':pd.DataFrame({'column1':['basketball','futbol','hockey','skiing'],
                                               'column2':['cheese','salt','pepper','paprika']})}
        self.test_dict_gold = ['four five six','spinach potato cauliflower','cheese salt pepper paprika']

    def test_convert_dict_to_list(self):
        self.assertListEqual([ag.df_slice_to_string(self.test_dict[key]) for key in self.test_dict],self.test_dict_gold)
        self.assertEqual(len(self.test_dict),len([ag.df_slice_to_string(self.test_dict[key]) for key in self.test_dict]))
        with self.assertRaises(KeyError):
            ag.df_slice_to_string(self.test_dict['key4'])

