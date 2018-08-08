/*
 * mx_import.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDE_MX_PORT_H_
#define _INCLUDE_MX_PORT_H_

#include <stdint.h>
#include <stddef.h>



/**
 * Mechanica is built with single precision in mind. In the future,
 * maybe there could be a need for double precision
 */
typedef float mx_real;

typedef size_t         Mx_ssize_t;



/* Declarations for symbol visibility.

  MxAPI_FUNC(type): Declares a public Mechanica API function and return type
  MxAPI_DATA(type): Declares public Mechanica data and its type
  MxMODINIT_FUNC:   A Mechanica module init function.  If these functions are
                    inside the Mechanica core, they are private to the core.
                    If in an extension module, it may be declared with
                    external linkage depending on the platform.

  As a number of platforms support/require "__declspec(dllimport/dllexport)",
  we support a HAVE_DECLSPEC_DLL macro to save duplication.
*/

/*
  All windows ports, except cygwin, are handled in PC/pyconfig.h.

  Cygwin is the only other autoconf platform requiring special
  linkage handling and it uses __declspec().
*/
#if defined(__CYGWIN__)
#       define HAVE_DECLSPEC_DLL
#endif

/* only get special linkage if built as shared or platform is Cygwin */
#if defined(Mx_ENABLE_SHARED) || defined(__CYGWIN__)
#       if defined(HAVE_DECLSPEC_DLL)
#               ifdef Mx_BUILD_CORE
#                       define MxAPI_FUNC(RTYPE) __declspec(dllexport) RTYPE
#                       define MxAPI_DATA(RTYPE) extern __declspec(dllexport) RTYPE
        /* module init functions inside the core need no external linkage */
        /* except for Cygwin to handle embedding */
#                       if defined(__CYGWIN__)
#                               define MxMODINIT_FUNC __declspec(dllexport) MxObject*
#                       else /* __CYGWIN__ */
#                               define MxMODINIT_FUNC MxObject*
#                       endif /* __CYGWIN__ */
#               else /* Mx_BUILD_CORE */
        /* Building an extension module, or an embedded situation */
        /* public Mechanica functions and data are imported */
        /* Under Cygwin, auto-import functions to prevent compilation */
        /* failures similar to those described at the bottom of 4.1: */
        /* http://docs.python.org/extending/windows.html#a-cookbook-approach */
#                       if !defined(__CYGWIN__)
#                               define MxAPI_FUNC(RTYPE) __declspec(dllimport) RTYPE
#                       endif /* !__CYGWIN__ */
#                       define MxAPI_DATA(RTYPE) extern __declspec(dllimport) RTYPE
        /* module init functions outside the core must be exported */
#                       if defined(__cplusplus)
#                               define MxMODINIT_FUNC extern "C" __declspec(dllexport) MxObject*
#                       else /* __cplusplus */
#                               define MxMODINIT_FUNC __declspec(dllexport) MxObject*
#                       endif /* __cplusplus */
#               endif /* Mx_BUILD_CORE */
#       endif /* HAVE_DECLSPEC */
#endif /* Mx_ENABLE_SHARED */

/* If no external linkage macros defined by now, create defaults */
#ifndef MxAPI_FUNC
#  ifdef __cplusplus
#    define MxAPI_FUNC(RTYPE) extern "C" RTYPE
#  else
#    define MxAPI_FUNC(RTYPE) extern RTYPE
#  endif
#endif

#ifndef MxAPI_DATA
#       define MxAPI_DATA(RTYPE) extern RTYPE
#endif
#ifndef MxMODINIT_FUNC
#       if defined(__cplusplus)
#               define MxMODINIT_FUNC extern "C" MxObject*
#       else /* __cplusplus */
#               define MxMODINIT_FUNC MxObject*
#       endif /* __cplusplus */
#endif

/** Macro for pre-defining opaque public data types */
#ifndef MxAPI_STRUCT
#    if defined(__cplusplus)
#        define MxAPI_STRUCT(TYPE) struct TYPE
#    else
#        define MxAPI_STRUCT(TYPE) typedef struct TYPE TYPE
#    endif
#endif


#ifndef WIN32


/**
 * Mechanica return code, same as Windows.
 *
 * To test an HRESULT value, use the FAILED and SUCCEEDED macros.

 * The high-order bit in the HRESULT or SCODE indicates whether the
 * return value represents success or failure.
 * If set to 0, SEVERITY_SUCCESS, the value indicates success.
 * If set to 1, SEVERITY_ERROR, it indicates failure.
 *
 * The facility field indicates from bits 26-16 is the system
 * service responsible for the error. The FACILITY_ITF = 4 is used for most status codes
 * returned from interface methods. The actual meaning of the
 * error is defined by the interface. That is, two HRESULTs with
 * exactly the same 32-bit value returned from two different
 * interfaces might have different meanings.
 *
 * The code field from bits 15-0 is the application defined error
 * code.
 */
typedef int32_t HRESULT;

#define S_OK             0x00000000 // Operation successful
#define E_ABORT          0x80004004 // Operation aborted
#define E_ACCESSDENIED   0x80070005 // General access denied error
#define E_FAIL           0x80004005 // Unspecified failure
#define E_HANDLE         0x80070006 // Handle that is not valid
#define E_INVALIDARG     0x80070057 // One or more arguments are not valid
#define E_NOINTERFACE    0x80004002 // No such interface supported
#define E_NOTIMPL        0x80004001 // Not implemented
#define E_OUTOFMEMORY    0x8007000E // Failed to allocate necessary memory
#define E_POINTER        0x80004003 // Pointer that is not valid
#define E_UNEXPECTED     0x8000FFFF // Unexpected failure


/**
 * Provides a generic test for success on any status value.
 * Parameters:
 * hr: The status code. A non-negative number indicates success.
 * Return value: TRUE if hr represents a success status value;
 * otherwise, FALSE.
 */
#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)

/**
 * Creates an HRESULT value from its component pieces.
 * Parameters
 * sev: The severity.
 * fac: The facility.
 * code: The code.
 * Return value: The HRESULT value.
 *
 * Note   Calling MAKE_HRESULT for S_OK verification carries a
 * performance penalty. You should not routinely use MAKE_HRESULT
 * for successful results.
 */
#define MAKE_HRESULT(sev,fac,code) \
    static_cast<HRESULT>((static_cast<uint32_t>(sev)<<31) | (static_cast<uint32_t>(fac)<<16) | (static_cast<uint32_t>(code)))

/**
 * Extracts the code portion of the specified HRESULT.
 * Parameters:
 * hr: The HRESULT value.
 * Return value: The code.
 */
#define HRESULT_CODE(hr)    ((hr) & 0xFFFF)

/**
 * Extracts the facility of the specified HRESULT.
 * Parameters:
 * hr: The HRESULT value.
 * Return value: The facility.
 */
#define HRESULT_FACILITY(hr)  (((hr) >> 16) & 0x1fff)

/**
 * Extracts the severity field of the specified HRESULT.
 * Parameters:
 * hr: The HRESULT.
 * Return value: The severity field.
 */
#define HRESULT_SEVERITY(hr)  (((hr) >> 31) & 0x1)


/**
 * debug verify an operation succeedes
 *
 */

#ifndef NDEBUG
#define VERIFY(hr) assert(SUCCEEDED(hr))
#else
#define VERIFY(hr) hr
#endif


#endif

/**
 * Error code faculties for Mechanica errors.
 */
#define FACULTY_MESH 10
#define FACULTY_MESHOPERATION 11


#endif /* _INCLUDE_MX_PORT_H_ */


