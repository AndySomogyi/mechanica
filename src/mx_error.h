/*
 * mx_error.h
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#ifndef SRC_MX_ERROR_H_
#define SRC_MX_ERROR_H_

#include "mechanica_private.h"

#define mx_error(code, msg) mx_set_error(code, msg, __LINE__, __FILE__, __PRETTY_FUNCTION__)


HRESULT mx_set_error(HRESULT code, const char* msg, int line, const char* file, const char* func);



#endif /* SRC_MX_ERROR_H_ */
