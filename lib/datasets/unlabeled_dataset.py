"""
Created by Fabio Hellmann.
"""

import os
from typing import Tuple

import numpy as np
from PIL import Image
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from lib.datasets import DatasetSplit
from lib.utils import glob_dir


class UnlabeledDataset(Dataset):
    """
    The UnlabeledDataset wraps a folder structure of a dataset with train, val, and test folders.
    """

    def __init__(self, data_dir: str, split: DatasetSplit, transforms: Compose):
        """
        Create a new UnlabeledDataset.
        @param data_dir: The root directory of the dataset.
        @param split: Which dataset split should be considered.
        @param transforms: The transformations that are applied to the images before returning.
        """
        data_dir = os.path.join(data_dir, split.value['folder'])
        logger.debug(f'Setup UnlabeledDataset on path: {data_dir}')
        # Search for all files available in the filesystem
        self.files_found = glob_dir(data_dir)
        self.transforms = transforms
        logger.info(f'Finished "{os.path.basename(data_dir)}" Dataset (total={len(self)}) Setup!')

    def __len__(self) -> int:
        return len(self.files_found)

    def __getitem__(self, idx) -> Tuple[Tensor, np.ndarray]:
        img = Image.open(self.files_found[idx])
        img = self.transforms(img)
        return img, np.asarray([-1])
