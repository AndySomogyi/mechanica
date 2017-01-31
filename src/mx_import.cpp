/*
 * ca_import.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: andy
 */

#include "mechanica_private.h"
#include <string>

using std::string;

using namespace std;

extern "C"
{

MxObject* MxImport_ImportModule(const char* name)
{
	return 0;
}

MxObject* MxImport_ImportModuleNoBlock(const char* name)
{
	return 0;
}

MxObject* MxImport_ImportModuleEx(char* name, MxObject* globals,
		MxObject* locals, MxObject* fromlist)
{
	return 0;
}

MxObject* MxImport_ImportModuleLevel(char* name, MxObject* globals,
		MxObject* locals, MxObject* fromlist, int level)
{
	return 0;
}

MxObject* MxImport_Import(MxObject* name)
{
	return 0;
}

MxObject* MxImport_ReloadModule(MxObject* m)
{
	return 0;
}

MxObject* MxImport_AddModule(const char* name)
{
	return 0;
}

MxObject* MxImport_ExecCodeModule(char* name, MxObject* co)
{
	return 0;
}

MxObject* MxImport_ExecCodeModuleEx(char* name, MxObject* co, char* pathname)
{
	return 0;
}

long MxImport_GetMagicNumber()
{
	return 0;
}

MxObject* MxImport_GetModuleDict()
{
	return 0;
}

}
