# GANonymization: A GAN-based Face Anonymization Framework for Preserving Emotional Expressions

[[Paper](https://arxiv.org/abs/2305.02143)]
[[Demo]()]


## Quick Start
The project was tested only with python version 3.8. Newer versions might work as well.

### Installation
1. Clone repository: `git clone https://github.com/hcmlab/GANonymization`.
2. Install using `setup.py`:
```bash
pip install -e .
```
or:
```bash
pip install git+https://github.com/hcmlab/GANonymization
```

### Training
You can either download a pre-trained GANonymization model:
* [Trained for 25 epochs](https://mediastore.rz.uni-augsburg.de/get/NsLjQYey65/) (publication version)
* [Trained for 50 epochs](https://mediastore.rz.uni-augsburg.de/get/Sfle_etB1D/)

Or train the GANonymization model yourself:
```bash
python main.py train_pix2pix --data_dir <data directory> --log_dir <log directory> --models_dir <models directory> --output_dir <output directory> --dataset_name <name of the dataset>
```

### Anonymize
You can run the anonymization as followed:
```bash
python main.py anonymize_image --model_file <path to model file> --input_file <image file> --output_file <output file>
```


## Citation
If you are using GANonymization in your research please consider giving us a citation:

```
@misc{hellmann2023ganonymization,
      title={GANonymization: A GAN-based Face Anonymization Framework for Preserving Emotional Expressions}, 
      author={Fabio Hellmann and Silvan Mertes and Mohamed Benouis and Alexander Hustinx and Tzung-Chien Hsieh and Cristina Conati and Peter Krawitz and Elisabeth André},
      year={2023},
      eprint={2305.02143},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## License
```
MIT License

Copyright (c) 2023 Chair of Human-Centered Artifical Intelligence, University of Augsburg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```