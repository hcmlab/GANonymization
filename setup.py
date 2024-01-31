"""
Created by Fabio Hellmann.
"""

from setuptools import setup, find_packages

MAJOR = 1
MINOR = 0
PATCH = 1

with open('requirements.txt', encoding='UTF-8') as f:
    required = f.read().splitlines()

setup(
    name="GANonymization",
    version=f"{MAJOR}.{MINOR}.{PATCH}",
    description="GANonymization: A GAN-based Face Anonymization Framework for Preserving "
                "Emotional Expressions",
    author="Fabio Hellmann, Silvan Mertes, Mohamed Benouis, Alexander Hustinx, Tzung-Chien "
           "Hsieh, Cristina Conati, Peter Krawitz, Elisabeth André",
    author_email="fabio.hellmann@informatik.uni-augsburg.de",
    url="https://github.com/hcmlab/GANonymization",
    packages=find_packages(),
    install_requires=required,
)
