from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="GANonymization",
    version="1.0.0",
    description="GANonymization: A GAN-based Face Anonymization Framework for Preserving Emotional Expressions",
    author="Fabio Hellmann, Silvan Mertes, Mohamed Benouis, Alexander Hustinx, Tzung-Chien Hsieh, Cristina Conati, Peter Krawitz, Elisabeth André",
    author_email="fabio.hellmann@informatik.uni-augsburg.de",
    url="https://github.com/hcmlab/GANonymization",
    packages=find_packages(),
    install_requires=required,
)
