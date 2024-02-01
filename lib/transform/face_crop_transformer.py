"""
Created by Fabio Hellmann.
"""
from typing import List

import numpy as np
from retinaface import RetinaFace


class FaceCrop:
    """
    The FaceCrop transformer is based on the RetinaFace library which extracts all available faces
    from the given image.
    """

    def __init__(self, align: bool):
        """
        @param align: Whether the face should be aligned.
        """
        self.align = align

    def __call__(self, pic: np.ndarray) -> List[np.ndarray]:
        """
        @param pic (numpy.ndarray): Image to be converted to cropped faces.
        @return: List[numpy.ndarray]: Converted image.
        """
        return [i[:, :, ::-1] for i in RetinaFace.extract_faces(pic, align=self.align)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
