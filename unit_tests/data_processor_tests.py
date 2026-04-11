import os
import unittest
from tcp_utils import Data_Processor as dp
import pandas as pd
import numpy as np
from collections import Counter
import tempfile,types

class CorpusTransformerTests(unittest.TestCase):

    def test_get_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_folder_path = os.path.join(tmpdir,'temp_folder')
            os.makedirs(temp_folder_path)

            with open(os.path.join(tmpdir, 'test_file.txt'), 'w') as f:
                f.write('This is a test file.')

            corpus_transformer = dp.Corpus_Transformer()
            output = corpus_transformer.get_subdirectories(tmpdir)
            self.assertTrue(len(output) == 1)


class CorpusLoaderTests(unittest.TestCase):
    def test_get_parallel_texts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_folder1_path = os.path.join(tmpdir,'temp_folder1')
            os.makedirs(temp_folder1_path)

            temp_folder2_path = os.path.join(tmpdir,'temp_folder2')
            os.makedirs(temp_folder2_path)

            with open(os.path.join(temp_folder1_path, 'test_file1.txt'), 'w') as f:
                f.write('This is a test file.')

            with open(os.path.join(temp_folder2_path, 'test_file1.txt'), 'w') as f:
                f.write('This is the parallel matching test file.')

            corpus_loader = dp.Corpus_Loader()
            test_dict = {'source_language':temp_folder1_path, 'target_language':temp_folder2_path}
            invalid_call = corpus_loader.get_parallel_text('test_file1.txt',temp_folder1_path,
                                                     temp_folder2_path)
            valid_call = corpus_loader.get_parallel_text('test_file1.txt',source_lang_path=temp_folder1_path,
                                                         target_lang_path=temp_folder2_path)
            with self.assertRaises(StopIteration):
                next(invalid_call)

            self.assertTrue(len(list(valid_call)) == 1)
            self.assertTrue(all(isinstance(item,tuple) for item in list(valid_call)))


    def test_corpus_loader_transform(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_folder1_path = os.path.join(tmpdir,'temp_folder1')
            os.makedirs(temp_folder1_path)

            temp_folder2_path = os.path.join(tmpdir,'temp_folder2')
            os.makedirs(temp_folder2_path)

            with open(os.path.join(temp_folder1_path, 'test_file1.txt'), 'w') as f:
                f.write('This is a test file.')

            with open(os.path.join(temp_folder2_path, 'test_file1.txt'), 'w') as f:
                f.write('This is the parallel matching test file.')

            corpus_loader = dp.Corpus_Loader()
            input = {'source_lang_path':temp_folder1_path, 'target_lang_path':temp_folder2_path}
            output = corpus_loader.fit_transform(input)
            self.assertTrue(isinstance(output,types.GeneratorType))
            self.assertTrue(isinstance(next(output),tuple))

