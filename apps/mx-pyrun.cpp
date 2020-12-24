/* Minimal main program -- everything is loaded from the library */


#ifdef __APPLE__
#define Py_BUILD_CORE
#endif
#include <iostream>
#include <string>

#include <Python.h>
#include <stddef.h>
#include "mx-pyrun.h"

//#include <string>

#ifdef MS_WINDOWS
int wmain(int argc, wchar_t **argv)
{
    return Py_Main(argc, argv);
}

#else

int main(int argc, char **argv)
{

   std::wstring pypath(Py_GetPath());

   std::wcout << L"path: " << pypath << std::endl;

   pypath = pypath + L":" + PY_SITEPACKAGES + L":" + MX_PYMODULE_DIR + L":" + NUMPY_PYPATH;

   Py_SetPath(pypath.c_str());

   std::wcout << "new path: " << Py_GetPath() << std::endl;





#ifdef __APPLE__
    return _Py_UnixMain(argc, argv);
#else
    Py_Initialize();

    wchar_t** _argv = (wchar_t**)malloc(sizeof(wchar_t*)*argc);

    for (int i=0; i<argc; i++) {
        wchar_t* arg = Py_DecodeLocale(argv[i], NULL);
        _argv[i] = arg;
    }
    int result =  Py_Main(argc, _argv);
    free(_argv);
    return result;
#endif
}

#endif
