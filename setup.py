import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nano-autograds",
    version="0.1.0",
    author="Youness EL BRAG",
    author_email="younsselbrag@gmail.com",
    description="A tinyTroch scalar-Engine Nano-autograd a Micro-Framework with a small PyTorch-like neural network library on top.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deep-matter/Nano-AutoGrad/autograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)