/*
 * ca_runtime.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDE_CA_RUNTIME_H_
#define _INCLUDE_CA_RUNTIME_H_

#include <carbon.h>
#include <stdio.h>


/**
 * Initialize the entire runtime.
 */
CAPI_FUNC(HRESULT) Mx_Initialize(int);


CAPI_FUNC(void) Mx_Finalize(void);


CAPI_FUNC(void) Mx_Exit(int);


#endif /* _INCLUDE_CA_RUNTIME_H_ */
