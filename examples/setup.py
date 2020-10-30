import os
from setuptools import setup
from setuptools import Extension
from setuptools.dist import Distribution
import re

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

# get the version info
def version():
    s = open("VERSION.txt").read()
    major = re.search("VERSION_MAJOR\s+([0-9]*)", s).groups()[0]
    minor = re.search("VERSION_MINOR\s+([0-9]*)", s).groups()[0]
    patch = re.search("VERSION_PATCH\s+([0-9]*)", s).groups()[0]
    dev =   re.search("VERSION_DEV\s+([0-9]*)", s).groups()[0]

    ver = major + "." + minor + "." + patch
    if len(dev) > 0:
        ver = ver + ".dev" + dev

    print("making version: ", ver)
    return ver




setup(
    name = "mechanica",
    version = version(),
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
        "Programming Language :: Python :: 3.6",
    ],
    package_data = { '' : ['*.so', '*.dll', '*.pyd'], 'mechanica' : ['examples/*.*']},
    python_requires='>=3.7.3',
    install_requires=[
        'numpy>=1.19.1'
    ],
    distclass=BinaryDistribution
)
