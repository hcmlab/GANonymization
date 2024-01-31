"""
Created by Fabio Hellmann.
"""

from setuptools import setup, find_packages

major = 1
minor = 0
patch = 1

with open('requirements.txt', encoding='UTF-8') as f:
    required = f.read().splitlines()

setup(
    name="GANonymization",
    version=f"{major}.{minor}.{patch}",
    description="GANonymization: A GAN-based Face Anonymization Framework for Preserving "
                "Emotional Expressions",
    author="Fabio Hellmann, Silvan Mertes, Mohamed Benouis, Alexander Hustinx, Tzung-Chien "
           "Hsieh, Cristina Conati, Peter Krawitz, Elisabeth André",
    author_email="fabio.hellmann@informatik.uni-augsburg.de",
    url="https://github.com/hcmlab/GANonymization",
    packages=find_packages(),
    install_requires=required,
)
