/*
 * ca_module.h
 *
 *  Created on: Jul 6, 2015
 *      Author: andy
 *
 * module functions, definitions and documentation copied from official
 * python website for python compatiblity.
 */

#ifndef _INCLUDED_CA_MODULE_H_
#define _INCLUDED_CA_MODULE_H_

#include "mx_object.h"

MxAPI_STRUCT(MxModule);

#ifdef __cplusplus
extern "C"
{
#endif



/**
 * This instance of MxTypeObject represents the Mechanica module type. 
 */
MxAPI_DATA(struct MxType*) MxModule_Type;

/**
 * Return true if p is a module object, or a subtype of a module object.
 */
MxAPI_FUNC(int) MxModule_Check(MxObject *p);

/**
 * Return true if p is a module object, but not a subtype of MxModule_Type.
 */
MxAPI_FUNC(int) MxModule_CheckExact(MxObject *p);

/**
 * Return value: New reference.
 * Return a new module object with the __name__ attribute set to name.
 * Only the module’s __doc__ and __name__ attributes are filled in;
 * the caller is responsible for providing a __file__ attribute.
 */
MxAPI_FUNC(MxModule*) MxModule_New(const char *name);

/**
 * Return value: Borrowed reference.
 * Return the dictionary object that implements module‘s namespace;
 * this object is the same as the __dict__ attribute of the module object.
 * This function never fails. It is recommended extensions use other MxModule_*()
 * and MxObject_*() functions rather than directly manipulate a module’s __dict__.
 */
MxAPI_FUNC(MxObject*) MxModule_GetDict(MxModule *module);

/**
 * Return module‘s __name__ value. If the module does not provide one,
 * or if it is not a string, SystemError is raised and NULL is returned.
 */
MxAPI_FUNC(const char*) MxModule_GetName(MxModule *module);

/**
 * Return the name of the file from which module was loaded using module‘s __file__
 * attribute. If this is not defined, or if it is not a string, raise SystemError
 * and return NULL.
 */
MxAPI_FUNC(const char*) MxModule_GetFilename(MxModule *module);

/**
 * Add an object to module as name. This is a convenience function which can be used
 * from the module’s initialization function. This steals a reference to value.
 * Return -1 on error, 0 on success.
 */
MxAPI_FUNC(int) MxModule_AddObject(MxModule *module, const char *name, MxObject *value);





#ifdef __cplusplus
}
#endif

#endif /* PYCALC_INCLUDE_CA_MODULE_H_ */
