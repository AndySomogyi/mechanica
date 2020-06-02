import os
from setuptools import setup
from setuptools import Extension
from setuptools.dist import Distribution

# Tested with wheel v0.29.0
class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "mechanica",
    version = "0.0.1.a1.dev5",
    author = "Andy Somogyi",
    author_email = "andy.somogyi@gmail.com",
    description = ("Interactive physics simulation engine"),
    license = "LGPL",
    keywords = "physics, molecular dynamics, center model, sub-cellular element",
    platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    url = "https://mechanica.readthedocs.io",
    packages=['mechanica'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
        "Programming Language :: Python :: 3.7",
    ],
    package_data = { '' : ['*.so']},
    python_requires='>=3.7.0',
    distclass=BinaryDistribution
)
