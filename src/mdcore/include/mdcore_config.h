/*
 * config.h
 *
 *  Created on: Mar 19, 2017
 *      Author: Andy Somogyi
 *
 * PRIVATE config include.
 * mdcore is set up to build multiple different version of the libray
 * from the same soruce using different preprocessor macros.
 *
 * The macros are enabled/disabled via cmake, and cmake generates the
 * appropriate public include file.
 */

#ifndef SRC_CONFIG_H_
#define SRC_CONFIG_H_

#ifdef MDCORE_SINGLE
#include "mdcore_single_config.h"
#endif

#include "platform.h"

#endif /* SRC_CONFIG_H_ */
