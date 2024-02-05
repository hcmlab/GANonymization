<div align="center">

______________________________________________________________________
# GANonymization

#### A GAN-based Face Anonymization Framework for Preserving Emotional Expressions
______________________________________________________________________

[![PyLint-Tests](https://github.com/hcmlab/GANonymization/actions/workflows/pylint.yml/badge.svg?branch=main&event=push)](https://github.com/hcmlab/GANonymization/actions/workflows/pylint.yml)

[**Paper**](https://dl.acm.org/doi/10.1145/3641107)
|
[**Demo**](https://hcmlab.github.io/GANonymization/)

</div>

> We introduce GANonymization, a novel face anonymization framework with facial expression-preserving abilities. Our approach is based on a high-level representation of a face which is synthesized into an anonymized version based on a generative adversarial network (GAN). The effectiveness of the approach was assessed by evaluating its performance in removing identifiable facial attributes to increase the anonymity of the given individual face. Additionally, the performance of preserving facial expressions was evaluated on several affect recognition datasets and outperformed the state-of-the-art method in most categories. Finally, our approach was analyzed for its ability to remove various facial traits, such as jewelry, hair color, and multiple others. Here, it demonstrated reliable performance in removing these attributes.

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
* [Trained for 50 epochs](https://mediastore.rz.uni-augsburg.de/get/Sfle_etB1D/) (demo version)

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
@article{10.1145/3641107,
      author = {Hellmann, Fabio and Mertes, Silvan and Benouis, Mohamed and Hustinx, Alexander and Hsieh, Tzung-Chien and Conati, Cristina and Krawitz, Peter and Andr\'{e}, Elisabeth},
      title = {GANonymization: A GAN-based Face Anonymization Framework for Preserving Emotional Expressions},
      year = {2024},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      issn = {1551-6857},
      url = {https://doi.org/10.1145/3641107},
      doi = {10.1145/3641107},
      abstract = {In recent years, the increasing availability of personal data has raised concerns regarding privacy and security. One of the critical processes to address these concerns is data anonymization, which aims to protect individual privacy and prevent the release of sensitive information. This research focuses on the importance of face anonymization. Therefore, we introduce GANonymization, a novel face anonymization framework with facial expression-preserving abilities. Our approach is based on a high-level representation of a face, which is synthesized into an anonymized version based on a generative adversarial network (GAN). The effectiveness of the approach was assessed by evaluating its performance in removing identifiable facial attributes to increase the anonymity of the given individual face. Additionally, the performance of preserving facial expressions was evaluated on several affect recognition datasets and outperformed the state-of-the-art methods in most categories. Finally, our approach was analyzed for its ability to remove various facial traits, such as jewelry, hair color, and multiple others. Here, it demonstrated reliable performance in removing these attributes. Our results suggest that GANonymization is a promising approach for anonymizing faces while preserving facial expressions.},
      note = {Just Accepted},
      journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
      month = {jan},
      keywords = {face anonymization, emotion recognition, data privacy, emotion preserving, facial landmarks}
}
```

## License

```
MIT License

Copyright (c) 2023 Chair of Human-Centered Artificial Intelligence, University of Augsburg

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
