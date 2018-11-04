/*
 * CylinderComponent.h
 *
 *  Created on: Nov 02, 2018
 *      Author: andy
 */



#ifndef CYLINDERCOMPONENT_H_
#define CYLINDERCOMPONENT_H_

#include "mx_port.h"


MxAPI_STRUCT(CylinderTest);

#define CYL_EXPORT __declspec(dllexport)


 MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_Create(int32_t width, int32_t height, CylinderTest **result);

 MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_Draw(CylinderTest *comp);

 MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_Step(CylinderTest *comp, float step);

 MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_LoadMesh(CylinderTest *comp, const char* path);

 MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_GetScalarValue(CylinderTest *comp, uint32_t id, float* result);

 MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_SetScalarValue(CylinderTest *comp, uint32_t id, float value);

#endif /* CYLINDERCOMPONENT_H_ */
