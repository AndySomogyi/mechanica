Getting Mechanica
=================



The easiest way to install Mechanica for most users is via PIP. We presently
have PIP binaries for Windows and MacOS. We do have a Ubuntu 18.04 pip wheels,
available on GitHub, but these are not on PyPi because they are not
"manylinux" compatible. 

Mechanica requires at least Python version 3.7.0, on Mac you can install Python
a variety of ways, but we typicaly use Brew, `<https://brew.sh>`_.


.. warning::
   Mechanica DOES NOT WORK CORRECTLY IN SPYDER ON WINDOWS. Spyder uses a
   multi-process architecture, where user code is in a separate process, and
   there is a bug with trying to re-show a Mechanica window after it has been
   closed.

   This bug has only been seen in Spyder.

   You can run a Mechanica script in Spyder, just be aware that closing the
   window, followed by a ``m.show()`` will crash Python.

   We reccomend the Jupyter Console *Qt Console* or *ipython* command line
   console, or straight python  until we can resolve this bug. 
   

    
.. _pip-install:

Installing via pip for Mac and Windows
--------------------------------------

*Note*, we presently only have Windows and Mac PyPi packages. 

Python comes with an inbuilt package management system,
`pip <https://pip.pypa.io/en/stable>`_. Pip can install, update, or delete
any official package. The PyPi home page for mechancia is

`<https://pypi.org/project/mechanica/>`_.

You can install packages via the command line by entering::

 python -m pip install --user mechanica

We recommend using an *user* install, sending the ``--user`` flag to pip.
``pip`` installs packages for the local user and does not write to the system
directories.


Installing via pip for Linux
----------------------------

We presently only support Ubuntu 18.04, pip wheel binaries are here:

`<https://github.com/AndySomogyi/mechanica/releases/download/0.0.4/mechanica-0.0.4.dev1-cp36-cp36m-linux_x86_64.whl>`_.

Download and run pip directly on this file. 

Preferably, do not use ``sudo pip``, as this combination can cause problems.

