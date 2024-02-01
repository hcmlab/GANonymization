"""
Created by Fabio Hellmann.
"""

import cv2
import mediapipe
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec, WHITE_COLOR


class FacialLandmarks478:
    """
    Extract 468 facial landmark points from the picture and return it in a 2-dimensional picture.
    """

    def __call__(self, pic: np.ndarray) -> np.ndarray:
        """
        @param pic (numpy.ndarray): Image to be converted to a facial landmark image
        with 468 points.
        @return: numpy.ndarray: Converted image.
        """
        point_image = np.zeros(pic.shape, np.uint8)
        with mediapipe.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks is not None and len(results.multi_face_landmarks) > 0:
                mediapipe.solutions.drawing_utils.draw_landmarks(
                    image=point_image,
                    landmark_list=results.multi_face_landmarks[0],
                    landmark_drawing_spec=DrawingSpec(color=WHITE_COLOR, thickness=1,
                                                      circle_radius=0))
        return point_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
