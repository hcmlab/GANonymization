"""
Created by Fabio Hellmann.
"""

import numpy as np
import torch
from torchvision.transforms import transforms

from lib.models.pix2pix import Pix2Pix


class Pix2PixTransformer:
    """
    The GANonymization transformer synthesizes images based on facial landmarks images.
    """

    def __init__(self, model_file: str, img_size: int, device: int):
        """
        Initialize the transformer and load pretrained model.
        @param model_file: The model file to load.
        @param img_size: The image size to be used for the model.
        @param device: The device to use to run the model's inference on.
        """
        self.model = Pix2Pix.load_from_checkpoint(checkpoint_path=model_file)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __call__(self, pic: np.ndarray) -> np.ndarray:
        """
        @param pic (numpy.ndarray): Image to be converted to a face image.
        @return: numpy.ndarray: Converted image.
        """
        pic_transformed = self.transforms_(pic)
        pic_transformed_device = pic_transformed.to(self.device)
        pic_transformed_device_batched = torch.unsqueeze(pic_transformed_device, dim=0)
        return self.model(pic_transformed_device_batched).detach().cpu().numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
