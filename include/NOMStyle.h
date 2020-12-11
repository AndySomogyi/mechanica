/*
 * nom_style.h
 *
 *  Created on: Jul 30, 2020
 *      Author: andy
 */

#ifndef INCLUDE_NOMSTYLE_H_
#define INCLUDE_NOMSTYLE_H_

#include <c_port.h>

CAPI_STRUCT(NOMStyle);

enum StyleFlags {
    STYLE_VISIBLE =    1 << 0,
    STYLE_STACKALLOC = 1 << 1
};

CAPI_FUNC(HRESULT) NOMStyle_SetColor(NOMStyle *s, PyObject *o);

CAPI_FUNC(HRESULT) NOMStyle_SetFlag(NOMStyle *s, StyleFlags flag, bool value);

CAPI_FUNC(NOMStyle*) NOMStyle_New(PyObject *args, PyObject *kwargs);

CAPI_FUNC(NOMStyle*) NOMStyle_Clone(NOMStyle* s);


/**
 * The the NOMStyle type type
 */
CAPI_DATA(PyTypeObject) NOMStyle_Type;





#endif /* INCLUDE_NOMSTYLE_H_ */
