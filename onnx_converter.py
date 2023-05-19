import os

import fire
import torch

from lib.models.pix2pix import Pix2Pix

ONNX_MODEL = os.path.join('models', 'GANonymization.onnx')


def convert_to_onnx(model_ckpt: str):
    model = Pix2Pix.load_from_checkpoint(model_ckpt, map_location='cpu')

    # Generate some random input data for the generator
    input_shape = (1, 3, 512, 512)  # Assuming input shape of (batch_size, channels, height, width)
    input_data = torch.randn(input_shape)

    # Export the generator to ONNX
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    torch.onnx.export(model.generator, input_data, ONNX_MODEL, verbose=True, opset_version=12,
                      output_names=['output'], dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    fire.Fire(convert_to_onnx)
