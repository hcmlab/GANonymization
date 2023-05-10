import cv2
from head_segmentation import HumanHeadSegmentationPipeline


class FaceSegmentation:
    """
    The FaceSegmentation transformer eliminates everything besides the face in the image.
    """

    def __call__(self, pic):
        """
        @param pic (PIL Image or numpy.ndarray): Image to be converted to a face segmentation.
        @return: numpy.ndarray: Converted image.
        """
        segmentation_pipeline = HumanHeadSegmentationPipeline()
        face_mask = segmentation_pipeline.predict(pic)
        segmented_region = pic * cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
        return segmented_region

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
