import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NetGrad",
    version="0.1.0",
    author="Hafsa metmari",
    author_email="hafsamitmari@gmail.com",
    description="A tiny scalar-valued autograd engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deep-matter/netGrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)