"""
Created by Fabio Hellmann.
"""

import os.path
import shutil
import urllib.request
import zipfile
from unittest import TestCase

from main import preprocess, anonymize_directory

TEST_DATA_DIR = 'tmp'


class SystemTest(TestCase):
    """
    Test cases for main.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(TEST_DATA_DIR, 'data')
        os.makedirs(cls.data_dir, exist_ok=True)
        cls.models_dir = os.path.join(TEST_DATA_DIR, 'models')
        os.makedirs(cls.models_dir, exist_ok=True)
        # Download Dataset
        tiny_celeba = urllib.request.urlretrieve(
            'https://mediastore.rz.uni-augsburg.de/get/fO0mqJhgjo/',
            os.path.join(cls.data_dir, 'tiny_celeba.zip'))
        with zipfile.ZipFile(tiny_celeba[0], 'r') as zip_ref:
            zip_ref.extractall(cls.data_dir)
        cls.celeba_dir = os.path.join(cls.data_dir, 'CelebA', 'celeba')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TEST_DATA_DIR)

    def test_preprocess(self):
        """
        Test the preprocessing function of main.py.
        """
        try:
            preprocess(self.celeba_dir)
        except RuntimeError as e:
            # Check for no runtime error
            self.fail(f'Unexpected Exception raised: {e}')

    def test_anonymization_25(self):
        """
        Test the anonymize_directory function of main.py.
        """
        try:
            # Download Model - 25 epochs
            model_25_ckpt = urllib.request.urlretrieve(
                "https://mediastore.rz.uni-augsburg.de/get/NsLjQYey65/",
                os.path.join(self.models_dir, "GANonymization_25.ckpt"))
            self.assertTrue(os.path.exists(model_25_ckpt[0]),
                            'Could not download GANonymization_25.ckpt')
            # Anonymize files
            anonymize_directory(model_25_ckpt[0], self.celeba_dir,
                                os.path.join(TEST_DATA_DIR, 'anonymized_25'))
        except RuntimeError as e:
            # Check for no runtime error
            self.fail(f'Unexpected Exception raised: {e}')

    def test_anonymization_50(self):
        """
        Test the anonymize_directory function of main.py.
        """
        try:
            # Download Model - 50 epochs
            model_50_ckpt = urllib.request.urlretrieve(
                "https://mediastore.rz.uni-augsburg.de/get/Sfle_etB1D/",
                os.path.join(self.models_dir, "GANonymization_50.ckpt"))
            self.assertTrue(os.path.exists(model_50_ckpt[0]),
                            'Could not download GANonymization_50.ckpt')
            # Anonymize files
            anonymize_directory(model_50_ckpt[0], self.celeba_dir,
                                os.path.join(TEST_DATA_DIR, 'anonymized_50'))
        except RuntimeError as e:
            # Check for no runtime error
            self.fail(f'Unexpected Exception raised: {e}')
