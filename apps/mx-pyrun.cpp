/* Minimal main program -- everything is loaded from the library */


#ifdef __APPLE__
#define Py_BUILD_CORE
#endif

#include <iostream>
#include <string>

#include <Python.h>
#include <stddef.h>
#include "mx-pyrun.h"

#if defined(WIN32)
#include <direct.h>
#define getcwd(buffer, maxlen) _getcwd(buffer, maxlen)
#endif

#if defined(__ARM_NEON)
wchar_t * Py_GetPath(void) {
    return NULL;
}

void Py_SetPath(const wchar_t *) {
}

int Py_BytesMain(int argc, char **argv) {
    return 0;
}
#endif


void print_cwd() {
    
    char* buffer = getcwd( NULL, 0 );
    
     // Get the current working directory:
     if( buffer == NULL )
        perror( "getcwd error" );
     else
     {
        std::cout << "mx-pyrun cwd: " << buffer << std::endl;
        free(buffer);
     }
}

//#include <string>

#ifdef MS_WINDOWS
int wmain(int argc, wchar_t **argv)
{
    void print_cwd();
    return Py_Main(argc, argv);
}

#else

    int main(int argc, char **argv)
    {
        print_cwd();

        std::wstring pypath(Py_GetPath());

        std::wcout << L"path: " << pypath << std::endl;

        pypath = pypath + L":" + PY_SITEPACKAGES + L":" + MX_PYMODULE_DIR + L":" + NUMPY_PYPATH;

        Py_SetPath(pypath.c_str());

        std::wcout << "new path: " << Py_GetPath() << std::endl;
        
    #if (PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 8)
        return Py_BytesMain(argc, argv);
    #else
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
    #endif
    }

#endif
