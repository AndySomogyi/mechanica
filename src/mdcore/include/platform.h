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

/* Get the inlining right. */
#ifndef INLINE
# if __GNUC__ && !__GNUC_STDC_INLINE__
#  define MX_INLINE extern inline
# else
#  define MX_INLINE inline
# endif
#endif


#if defined(__cplusplus)
#define	MDCORE_BEGIN_DECLS	extern "C" {
#define	MDCORE_END_DECLS	}
#else
#define	MDCORE_BEGIN_DECLS
#define	MDCORE_END_DECLS
#endif


#ifndef __has_attribute         // Optional of course.
  #define __has_attribute(x) 0  // Compatibility with non-clang compilers.
#endif


#if __has_attribute(always_inline)
#define MX_ALWAYS_INLINE __attribute__((always_inline)) MX_INLINE
#else
#define MX_ALWAYS_INLINE MX_INLINE
#endif



#endif /* INCLUDE_PLATFORM_H_ */
