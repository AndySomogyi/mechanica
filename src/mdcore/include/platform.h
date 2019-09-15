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

#if defined(__cplusplus)
#define	MDCORE_BEGIN_DECLS	extern "C" {
#define	MDCORE_END_DECLS	}
#else
#define	MDCORE_BEGIN_DECLS
#define	MDCORE_END_DECLS
#endif


#endif /* INCLUDE_PLATFORM_H_ */
