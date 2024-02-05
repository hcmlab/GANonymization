"""
Created by Fabio Hellmann.
"""
import urllib.request

import cv2
import numpy as np
from head_segmentation import HumanHeadSegmentationPipeline, HEAD_SEGMENTATION_MODEL_PATH


class FaceSegmentation:
    """
    The FaceSegmentation transformer eliminates everything besides the face in the image.
    """
    MODEL_URL = 'https://mediastore.rz.uni-augsburg.de/get/SFRX3tkMpT/'

    def __init__(self):
        urllib.request.urlretrieve(self.MODEL_URL, filename=HEAD_SEGMENTATION_MODEL_PATH)
        self.segmentation_pipeline = HumanHeadSegmentationPipeline()

    def __call__(self, pic: np.ndarray) -> np.ndarray:
        """
        @param pic (numpy.ndarray): Image to be converted to a face segmentation.
        @return: numpy.ndarray: Converted image.
        """
        face_mask = self.segmentation_pipeline.predict(pic)
        segmented_region = pic * cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
        return segmented_region

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
