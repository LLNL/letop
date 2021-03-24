# -*- coding: utf-8 -*-
#
import codecs
import os

from setuptools import find_packages, setup

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "lestofire", "__about__.py"), "rb") as f:
    exec(f.read(), about)


def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()


setup(
    name="lestofire",
    version=about["__version__"],
    packages=find_packages(),
    author=about["__author__"],
    author_email=about["__email__"],
    install_requires=['numpy>=1.12.1',
                      'scipy>=0.19.1',
                      'cvxopt>=1.2.1',
                      'nullspace-optimizer @ git+https://gitlab.com/salazardetroya/null-space-optimizer@lestofire'],
    description="A little bit of foobar in your life",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license=about["__license__"],
    classifiers=[
        about["__license__"],
        about["__status__"],
        # See <https://pypi.org/classifiers/> for all classifiers.
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3",
    entry_points={"console_scripts": ["lestofire-show = lestofire.cli:show"]},
)
