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
    def setUp(self):
        self.data_dir = os.path.join(TEST_DATA_DIR, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.models_dir = os.path.join(TEST_DATA_DIR, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        # Download Model - 25 epochs
        self.model_25_ckpt = urllib.request.urlretrieve(
            "https://mediastore.rz.uni-augsburg.de/get/NsLjQYey65/",
            os.path.join(self.models_dir, "GANonymization_25.ckpt"))
        self.assertTrue(os.path.exists(self.model_25_ckpt[0]),
                        f'Expected to download GANonymization_25.ckpt')
        # Download Model - 50 epochs
        self.model_50_ckpt = urllib.request.urlretrieve(
            "https://mediastore.rz.uni-augsburg.de/get/Sfle_etB1D/",
            os.path.join(self.models_dir, "GANonymization_50.ckpt"))
        self.assertTrue(os.path.exists(self.model_50_ckpt[0]),
                        f'Expected to download GANonymization_50.ckpt')
        # Download Dataset
        self.tiny_celeba = urllib.request.urlretrieve(
            'https://mediastore.rz.uni-augsburg.de/get/em2na80Ljb/',
            os.path.join(self.data_dir, 'tiny_celeba.zip'))
        self.assertTrue(os.path.exists(self.tiny_celeba[0]),
                        f'Expected to download tiny-celeba.zip')
        with zipfile.ZipFile(self.tiny_celeba[0], 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        self.celeba_dir = os.path.join(self.data_dir, 'CelebA', 'celeba')

    def test_preprocess(self):
        try:
            preprocess(self.celeba_dir)
        except RuntimeError as e:
            # Check for no runtime error
            self.fail(f'Unexcepted Exception raised: {e}')

    def test_anonymization(self):
        try:
            # Anonymize files
            anonymize_directory(self.model_25_ckpt[0], self.celeba_dir,
                                os.path.join(TEST_DATA_DIR, 'anonymized_25'))
            anonymize_directory(self.model_50_ckpt[0], self.celeba_dir,
                                os.path.join(TEST_DATA_DIR, 'anonymized_50'))
        except RuntimeError as e:
            # Check for no runtime error
            self.fail(f'Unexcepted Exception raised: {e}')

    def tearDown(self):
        shutil.rmtree(TEST_DATA_DIR)
