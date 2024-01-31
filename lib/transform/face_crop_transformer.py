"""
Created by Fabio Hellmann.
"""

import numpy as np
from retinaface import RetinaFace


class FaceCrop:
    """
    The FaceCrop transformer is based on the RetinaFace library which extracts all available faces
    from the given image.
    """

    def __init__(self, align: bool, multiple_faces: bool):
        """
        @param align: Whether the face should be aligned.
        @param multiple_faces: Whether multiple faces should be detected.
        """
        self.align = align
        self.multiple_faces = multiple_faces

    def __call__(self, pic: np.ndarray):
        """
        @param pic (numpy.ndarray): Image to be converted to cropped faces.
        @return: numpy.ndarray: Converted image.
        """
        faces = [i[:, :, ::-1] for i in RetinaFace.extract_faces(pic, align=self.align)]
        if self.multiple_faces:
            return faces
        elif len(faces) > 0:
            return faces[0]
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
