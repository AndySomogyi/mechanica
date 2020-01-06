/*
 * ca_import.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: andy
 */


#ifdef CType
#error CType is macro
#endif

#include "mechanica_private.h"


#ifdef CType
#error CType is macro
#endif

#include <string>

using std::string;

using namespace std;

extern "C"
{

CObject* MxImport_ImportModule(const char* name)
{
	return 0;
}

CObject* MxImport_ImportModuleNoBlock(const char* name)
{
	return 0;
}

CObject* MxImport_ImportModuleEx(char* name, CObject* globals,
		CObject* locals, CObject* fromlist)
{
	return 0;
}

CObject* MxImport_ImportModuleLevel(char* name, CObject* globals,
		CObject* locals, CObject* fromlist, int level)
{
	return 0;
}

CObject* MxImport_Import(CObject* name)
{
	return 0;
}

CObject* MxImport_ReloadModule(CObject* m)
{
	return 0;
}

CObject* MxImport_AddModule(const char* name)
{
	return 0;
}

CObject* MxImport_ExecCodeModule(char* name, CObject* co)
{
	return 0;
}

CObject* MxImport_ExecCodeModuleEx(char* name, CObject* co, char* pathname)
{
	return 0;
}

long MxImport_GetMagicNumber()
{
	return 0;
}

CObject* MxImport_GetModuleDict()
{
	return 0;
}

}
