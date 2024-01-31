"""
Created by Fabio Hellmann.
"""

import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from lib.utils import glob_dir


class ImageDataset(Dataset):
    """
    An ImageDataset manages the access to images in a folder structure of train, val, and test.
    """

    def __init__(self, root, transforms_=None, mode="train"):
        """
        Create a new ImageDataset.
        @param root: The root folder of the dataset.
        @param transforms_: The transformers to be applied to the image before the images are
        handed over.
        @param mode: The mode either "train", "val", or "test".
        """
        self.transform = Compose(transforms_)
        self.files = sorted(glob_dir(os.path.join(root, mode)))
        if mode == "train":
            self.files.extend(sorted(glob_dir(os.path.join(root, "test"))))

    def __getitem__(self, index):
        file_path = self.files[index % len(self.files)]
        img = Image.open(file_path)
        w, h = img.size
        img_a = img.crop((0, 0, int(w / 2), h))
        img_a = self.transform(img_a)
        img_b = img.crop((int(w / 2), 0, w, h))
        img_b = self.transform(img_b)
        return {"A": img_a, "B": img_b, "Filepath": file_path}

    def __len__(self):
        return len(self.files)
