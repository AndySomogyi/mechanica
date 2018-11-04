#include "CylinderComponent.h"
#include "CylinderModel.h"
#include "CylinderTest.h"

#include <iostream>




MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_Create(int32_t width, int32_t height, CylinderTest **result) {

	std::cout << "Hello from CylinderTest_Create(" << width << ", " << height << ")" << std::endl;
	//return S_OK;

	CylinderTest::Configuration conf;

	conf.setSize({ {width, height} });
	conf.setVersion(Magnum::GL::Version::GL330);

	CylinderTest *obj = new CylinderTest(conf);
	*result = obj;
	return S_OK;
}

MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_Draw(CylinderTest *comp) {
	comp->draw();
	return S_OK;
}

MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_Step(CylinderTest *comp, float step) {
	return E_NOTIMPL;
}

MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_LoadMesh(CylinderTest *comp, const char* path) {
	std::cout << "Hello from LoadMesh(" << path << ")" << std::endl;

	comp->loadModel(path);
	return S_OK;
}

MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_GetScalarValue(CylinderTest *comp, uint32_t id, float* result) {
	return E_NOTIMPL;
}

MxAPI_FUNC(HRESULT) CYL_EXPORT CylinderTest_SetScalarValue(CylinderTest *comp, uint32_t id, float value) {
	return E_NOTIMPL;
}
