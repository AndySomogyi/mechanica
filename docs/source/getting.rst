Getting Mechanica
=================



The easiest way to install Mechanica for most users is via PIP. We presently
have PIP binaries for MacOS, Linux should be ready within a week or so, and it
will take us another few weeks to create Windows binaries.

Mechanica requires at least Python version 3.7.7, on Mac you can install Python
a variety of ways, but we typicaly use Brew, `<https://brew.sh>`_. 

.. _pip-install:

Installing via pip
------------------


*Note*, we presently only have Mac PIP binaries. 

Python comes with an inbuilt package management system,
`pip <https://pip.pypa.io/en/stable>`_. Pip can install, update, or delete
any official package. The PyPi home page for mechancia is

`<https://pypi.org/project/mechanica/>`_.

You can install packages via the command line by entering::

 python -m pip install --user mechanica

We recommend using an *user* install, sending the ``--user`` flag to pip.
``pip`` installs packages for the local user and does not write to the system
directories. Preferably, do not use ``sudo pip``, as this combination can cause problems.

Pip accesses the Python Package Index, `PyPI <https://pypi.org/>`_ , which
stores almost 200,000 projects and all previous releases of said projects.
Because the repository keeps previous versions, you can pin to a version and
not worry about updates causing conflicts. Pip can also install packages in
local *virtualenv*, or virtual environment.
