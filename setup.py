from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyflux",
    version="0.1",
    description="Pupil Labs 3D Attention Flux Vizualization Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pupil-labs/pyflux",
    author="Pupil Labs",
    author_email="kai@pupil-labs.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
)
