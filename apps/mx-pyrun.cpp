/* Minimal main program -- everything is loaded from the library */


#ifdef __APPLE__
#define Py_BUILD_CORE
#endif
#include "Python.h"
#include <stddef.h>

#ifdef MS_WINDOWS
int wmain(int argc, wchar_t **argv)
{
    return Py_Main(argc, argv);
}

#else

int main(int argc, char **argv)
{
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
