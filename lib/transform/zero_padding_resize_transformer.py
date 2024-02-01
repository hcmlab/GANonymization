"""
Created by Fabio Hellmann.
"""

import numpy as np
from PIL import Image


class ZeroPaddingResize:
    """
    The ZeroPaddingResize transformer resizes the image into the given size and applies
    zero-padding to center the output image.
    """

    def __init__(self, size: int):
        """
        @param size: The size the image will be resized to.
        """
        self.size = size

    def __call__(self, pic: np.ndarray) -> np.ndarray:
        """
        @param pic (numpy.ndarray): Image to be converted to center zero-padded
        and resized image.
        @return: numpy.ndarray: Converted image.
        """
        if isinstance(pic, np.ndarray):
            pic = Image.fromarray(pic)
        if pic.width != self.size or pic.height != self.size:
            ratio = min(self.size / pic.width, self.size / pic.height)
            face_img = pic.resize((int(ratio * pic.width), int(ratio * pic.height)),
                                  Image.Resampling.LANCZOS)
            new_im = Image.new("RGB", (self.size, self.size))
            new_im.paste(face_img,
                         ((self.size - face_img.width) // 2, (self.size - face_img.height) // 2))
            pic = new_im
        return np.array(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
