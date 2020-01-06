/*
 * ca_module.cpp
 *
 *  Created on: Jul 6, 2015
 *      Author: andy
 */
#include "mechanica_private.h"


extern "C" {

MxModule* MxModule_New(const char* name)
{
    MX_NOTIMPLEMENTED
}

int MxModule_Check(CObject* p)
{
    MX_NOTIMPLEMENTED
}

int MxModule_CheckExact(CObject* p)
{
    MX_NOTIMPLEMENTED
}

const char* MxModule_GetName(MxModule* module)
{
    MX_NOTIMPLEMENTED
}

const char* MxModule_GetFilename(MxModule* module)
{
    MX_NOTIMPLEMENTED
}

int MxModule_AddObject(MxModule* module, const char* name, CObject* value)
{
    MX_NOTIMPLEMENTED
}

} // extern "C"

void MxModule_init(PyObject *m) {

}
