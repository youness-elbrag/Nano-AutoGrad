import setuptools
import os
import sys

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "."))
sys.path.insert(0, target_dir)

with open(target_dir +"/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nano-autograds",
    version="0.0.1",
    author="Youness EL BRAG",
    author_email="younsselbrag@gmail.com",
    description="A tinyTroch scalar-Engine Nano-autograd a Micro-Framework with a small PyTorch-like neural network library on top.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deep-matter/Nano-AutoGrad/tree/main/autograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)