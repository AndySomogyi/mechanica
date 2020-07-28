/*
 * platform.h
 *
 * Created on: Mar 21, 2017
 *     Author: Andy Somogyi
 *
 * Symbols and macros to supply platform-independent interfaces to basic
 * C language & library operations whose spellings vary across platforms.
 *
 * Macros for symbol export
 */

#ifndef INCLUDE_PLATFORM_H_
#define INCLUDE_PLATFORM_H_

#include "carbon.h"

#if defined(__cplusplus)
#define	MDCORE_BEGIN_DECLS	extern "C" {
#define	MDCORE_END_DECLS	}
#else
#define	MDCORE_BEGIN_DECLS
#define	MDCORE_END_DECLS
#endif


#ifdef _WIN32
#define _USE_MATH_DEFINES
#define bzero(b,len) memset((b), '\0', (len))
#else
#define algined_free(x) free(x)
#endif



#endif /* INCLUDE_PLATFORM_H_ */
