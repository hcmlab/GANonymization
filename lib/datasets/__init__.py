"""
Created by Fabio Hellmann.
"""

from enum import Enum


class DatasetSplit(Enum):
    """
    The split for a dataset and their specific folders.
    """
    TRAIN = {'value': 0, 'folder': 'train'}
    VALIDATION = {'value': 1, 'folder': 'val'}
    TEST = {'value': 2, 'folder': 'test'}
