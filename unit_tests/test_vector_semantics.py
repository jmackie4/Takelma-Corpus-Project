import unittest,types
from tcp_utils import vector_semantics as vs
import pandas as pd


class Test_Series_to_String(unittest.TestCase):
    def setUp(self):
        self.mixed_series = pd.Series(['1', '2',9,10,11])
        self.string_series = pd.Series(['this is the first string','this is the second one',
                                        'tbh,the first two words should be kept together'])
        self.int_series = pd.Series([1,2,3,4,5,6,7,8,9])
        self.int_list = [1,2,3,4]
        self.str_list = ['one','two','three','four']

    def test_series_to_str(self):
        self.assertTrue(isinstance(vs.convert_series_to_str(self.mixed_series),str))
        self.assertTrue(isinstance(vs.convert_series_to_str(self.string_series),str))
        self.assertTrue(isinstance(vs.convert_series_to_str(self.int_series),str))
        with self.assertRaises(AttributeError):
            vs.convert_series_to_str(self.int_list)
        with self.assertRaises(AttributeError):
            vs.convert_series_to_str(self.str_list)


class Test_Text_to_List(unittest.TestCase):
    def setUp(self):
        self.spaced_string = 'this is the first string'
        self.non_spaced_string = 'thisisthesecondone'
        self.int_list = [1,2,3,4,5,6,7,8,9]
        self.str_list = ['one','two','three','four']

    def test_text_to_list(self):
        self.assertTrue(isinstance(vs.convert_text_to_list(self.spaced_string),list))
        self.assertTrue(len(vs.convert_text_to_list(self.spaced_string)) == 5)
        self.assertTrue(len(vs.convert_text_to_list(self.non_spaced_string)) == 1)
        with self.assertRaises(AttributeError):
            vs.convert_text_to_list(self.int_list)
        with self.assertRaises(AttributeError):
            vs.convert_text_to_list(self.str_list)


class Test_Create_Ordered_Dict(unittest.TestCase):
    def setUp(self):
        self.ordered_list = ['a','b','c','d','e','f','a','b','c','d','e']
        self.shuffled_list = ['b','d','f','a','c','e','f','a','b','c','d']
        self.string = 'this is a test string'
        self.int_list = [1,2,3,4,5,6,7,8,9]
        self.shuffled_dict = {i:item for i,item in enumerate(set(self.shuffled_list))}
    def test_create_ordered_dict(self):
        self.assertTrue(isinstance(vs.create_ordered_dict_from_list(self.ordered_list),dict))
        self.assertDictEqual(vs.create_ordered_dict_from_list(self.ordered_list),vs.create_ordered_dict_from_list(self.shuffled_list))
        self.assertNotEqual(vs.create_ordered_dict_from_list(self.ordered_list),self.shuffled_dict)
        with self.assertRaises(AssertionError):
            vs.create_ordered_dict_from_list(self.string)


class Test_Create_Generator(unittest.TestCase):
    def setUp(self):
        self.string_series = pd.Series(['this is the first string','this is the second one',
                                        'tbh,the first two words should be kept together'])

    def test_create_generator(self):
        self.assertTrue(isinstance(vs.create_generator_from_series(self.string_series),types.GeneratorType))
        self.assertEqual(next(vs.create_generator_from_series(self.string_series)),self.string_series[2])








