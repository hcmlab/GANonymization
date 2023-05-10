from torchvision.transforms import transforms

from lib.models.pix2pix import Pix2Pix


class Pix2PixTransformer:
    """
    The GANonymization transformer synthesizes images based on facial landmarks 468 images.
    """

    def __init__(self, model_file: str, img_size: int, device: int):
        self.model = Pix2Pix.load_from_checkpoint(model_file)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.transforms_ = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __call__(self, pic):
        """
        @param pic (PIL Image or numpy.ndarray): Image to be converted to a face image.
        @return: numpy.ndarray: Converted image.
        """
        return self.model(self.transforms_(pic.to(self.device)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
