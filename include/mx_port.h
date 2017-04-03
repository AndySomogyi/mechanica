/*
 * mx_import.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDE_MX_IMPORT_H_
#define _INCLUDE_MX_IMPORT_H_

#include <stdint.h>
#include <stddef.h>

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


#endif /* _INCLUDE_CA_IMPORT_H_ */


