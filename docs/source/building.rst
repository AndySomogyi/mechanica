Setting Up a Development Enviorment
***********************************


Git
===

Git submodule updates


To update submodule, need to do both the followign:

git pull --recurse-submodules

git submodule update --recursive --remote


Windows
=======


Triplet: x64-windows-static

Cmake defaults to : Multi-threaded Debug DLL (/MDd)

but assimp built statically seems to be Multi-threaded Debug (/MTd)

vcpkg static linking

https://blog.iany.me/2017/03/vcpkg-static-linking/


Python Debug Builds
===================

It is very informative to run the Python interpreter in debug mode, especially
when developing advanced language extensions.


Numpy
-----
The default version of numpy, from PyPy does not work with debug builds of
Python, at least it does not on MacOS with Python 3.7 and Numpy version
1.18. The standard source build of Numpy will not work with debug Python builds,
confirmed with the above versions.

At least will debug builds, Numpy also will not work with the built in MacOS
'Accelerate' framework which provides a version of BLAS. Don't know exaclty why,
but it compiles and installs fine, but fails the sanity test::

   def _sanity_check():
      x = ones(2, dtype=float32)
      if not abs(x.dot(x) - 2.0) < 1e-5:
         raise AssertionError()

We don't know why the dot product fails with the Accelerate framework
lapack. The fix is to build Numpy with Open BLAS. This is simple, however poorly
documented. Steps:

1: Grab the numpy source code from git, and go into that directory::

   git clone https://github.com/your-user-name/numpy.git
   cd numpy
   git remote add upstream https://github.com/numpy/numpy.git

2: Install Open BLAS from Brew
::
   localhost:$ brew reinstall openblas

It will say something like:: 

   ==> Downloading ...
   ...
   ==> Pouring openblas-0.3.9.high_sierra.bottle.tar.gz
   ==> Caveats
       openblas is keg-only, which means it was not symlinked into /usr/local,
       because macOS provides BLAS and LAPACK in the Accelerate framework.

       For compilers to find openblas you may need to set:
          export LDFLAGS="-L/usr/local/opt/openblas/lib"
          export CPPFLAGS="-I/usr/local/opt/openblas/include"

       For pkg-config to find openblas you may need to set:
          export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig"

   ==> Summary
   🍺  /usr/local/Cellar/openblas/0.3.9: 23 files, 120MB


Make note of the ``LDFLAGS`` and ``CPPFLAGS`` values, you will need to paste these
into the Numpy config file.

3: In the numpy directory, there is a file called ``example.site.cfg``, rename
this to ``site.cfg``, and open it up with your favorite editor. Look for the line
that says ``openblas``, will look like this::
   # [openblas]
   # libraries = openblas
   # library_dirs = /opt/OpenBLAS/lib
   # include_dirs = /opt/OpenBLAS/include
   # runtime_library_dirs = /opt/OpenBLAS/lib

Uncomment these lines, and replace them with the values of ``LDFLAGS`` and
``CPPFLAGS`` from above::
   [openblas]
   libraries = openblas
   library_dirs = /usr/local/opt/openblas/lib
   include_dirs = /usr/local/opt/openblas/include
   runtime_library_dirs = /usr/local/opt/openblas/lib

4: The really important part is to tell the Numpy buuild to actually use Open
BLAS. There does not seem to be any way in the ``site.cfg`` file to tell numpy
this, rather it apears you have to use enviornment variables (ugh!!!). So, set
the ``NPY_BLAS_ORDER`` enviorment variable to use Open BLAS::

   export NPY_BLAS_ORDER=openblas

5: Then perform a configuration, build and install of Numpy with the following
commands (make sure to use your correct debug build Python here)::

   python3 setup.py config
   python3 setup.py build
   python3 setup.py install

Pay particular attention to the output of the config step::

   python3 setup.py config

It will look something like this::

   gcc: /var/folders/p7/vcwd91x50j96v_yh75xtl4p40000gn/T/tmpr2lbwhqr/source.c
   gcc /var/folders/p7/vcwd91x50j96v_yh75xtl4p40000gn/T/tmpr2lbwhqr/var/ ... 40000gn/T/tmpr2lbwhqr/a.out
   FOUND:
     libraries = ['openblas', 'openblas']
     library_dirs = ['/usr/local/opt/openblas/lib']
     language = c
     define_macros = [('HAVE_CBLAS', None)]
     runtime_library_dirs = ['/usr/local/opt/openblas/lib']

   FOUND:
     libraries = ['openblas', 'openblas']
     library_dirs = ['/usr/local/opt/openblas/lib']
     language = c
     define_macros = [('HAVE_CBLAS', None)]
     runtime_library_dirs = ['/usr/local/opt/openblas/lib']

     lapack_opt_info:
     lapack_mkl_info:
   FOUND:
     libraries = ['mkl_rt', 'pthread']
     library_dirs = ['/usr/local/lib']
     define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
     include_dirs = ['/usr/local/include', '/usr/include', '/Users/andy/local/include']

   FOUND:
     libraries = ['mkl_rt', 'pthread']
     library_dirs = ['/usr/local/lib']
     define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
     include_dirs = ['/usr/local/include', '/usr/include',
     '/Users/andy/local/include']

The first sections must say that it found `Open BLAS` first.

Then finish with the build and install steps. 
 
6: Test it `BOTH` from Python and IPython, very important. They seem to set up
different library paths, and frequently if numpy works in one, it won't work in
the other, if Open BLAS is not setup correctly. In IPython, simply::
   >>> import numpy

And test it from python via the command line as::

   python3 -c "import numpy; print(numpy.get_include())"


Debugging Mechanica From Python
-------------------------------

We provide the `mx-pyrun` app / projct. This is a trivial program that simply
calls the main Python `main` routine, but the purpose of this probram is to
serve as a target app, as an entry point in your IDE so that the mechanica
python library can be loaded and stepped through. 



Linux
=====


Prerequisites:

sudo apt-get install libjpeg-dev

sudo apt-get install python-dev

sudo apt-get install python3-pip
pip3 install ipython --user

pip3 install numpy --user


libgl1-mesa-dev

libassimp-dev
libgl1-mesa-dev
libxrandr-dev
libxinerama-dev
libxcursor-dev
libgegl-dev
libglfw3-dev

libxi-dev
