import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from lib.utils import glob_dir


class ImageDataset(Dataset):
    """
    An ImageDataset manages the access to images in a folder structure of train, val, and test.
    """

    def __init__(self, root, transforms_=None, mode="train"):
        """
        Create a new ImageDataset.
        @param root: The root folder of the dataset.
        @param transforms_: The transformers to be applied to the image before the images are handed over.
        @param mode: The mode either "train", "val", or "test".
        """
        # Set True if training data shall be mirrored for augmentation purposes:
        self._MIRROR_IMAGES = False
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob_dir(os.path.join(root, mode)))
        self.files.extend(sorted(glob_dir(os.path.join(root, mode))))
        if mode == "train":
            self.files.extend(sorted(glob_dir(os.path.join(root, "test"))))
            self.files.extend(sorted(glob_dir(os.path.join(root, "test"))))

    def __getitem__(self, index):
        file_path = self.files[index % len(self.files)]
        img = Image.open(file_path)
        w, h = img.size
        img_A = img.crop((0, 0, int(w / 2), h))
        img_B = img.crop((int(w / 2), 0, w, h))
        if self._MIRROR_IMAGES:
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B, "Filepath": file_path}

    def __len__(self):
        return len(self.files)
