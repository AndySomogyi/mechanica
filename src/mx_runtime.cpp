/*
 * ca_runtime.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: andy
 */



#include "mechanica_private.h"

int Mx_InteractiveFlag = 0;


CObject* MxErr_Occurred(void)
{
	return NULL;
}



void Mx_Finalize(void)
{

}

int Mx_IsInitialized(void)
{
	return 0;
}

void MxErr_Print(void)
{
}

void MxErr_PrintEx(int int1)
{
}

int MxRun_AnyFileExFlags(FILE* fp, const char* filename, int closeit,
		MxCompilerFlags* flags)
{
    MX_NOTIMPLEMENTED
}


/*
 * The file descriptor fd is considered ``interactive'' if either
 *   a) isatty(fd) is TRUE, or
 *   b) the -i flag was given, and the filename associated with
 *      the descriptor is NULL or "<stdin>" or "???".
 */
int Mx_FdIsInteractive(FILE* fp, const char* filename)
{
    MX_NOTIMPLEMENTED
}

int MxRun_InteractiveOne(FILE* fp, const char* filename)
{
    MX_NOTIMPLEMENTED
}

int MxRun_InteractiveOneFlags(FILE *fp, const char *filename, MxCompilerFlags *flags)
{
	MX_NOTIMPLEMENTED
}

int MxRun_InteractiveLoop(FILE *f, const char *p)
{
    MX_NOTIMPLEMENTED
}

int MxRun_InteractiveLoopFlags(FILE *fp, const char *filename, MxCompilerFlags *flags)
{
    MX_NOTIMPLEMENTED
}

CObject* MxSys_GetObject(const char* name)
{
	MX_NOTIMPLEMENTED
}

FILE* MxSys_GetFile(const char* name, FILE* def)
{
	MX_NOTIMPLEMENTED
}

int MxSys_SetObject(const char* name, CObject* v)
{
	MX_NOTIMPLEMENTED
}

int MxRun_SimpleStringFlags(const char* command, MxCompilerFlags* flags)
{
    MX_NOTIMPLEMENTED
}

int MxRun_SimpleFileExFlags(FILE* fp, const char* filename, int closeit,
		MxCompilerFlags* flags)
{
	MX_NOTIMPLEMENTED
}


CObject* MxRun_StringFlags(const char* str, int start, CObject* globals, CObject* locals, MxCompilerFlags* flags)
{
	MX_NOTIMPLEMENTED
}

CObject* MxRun_FileExFlags(FILE* fp, const char* filename, int start, CObject* globals, CObject* locals, int closeit, MxCompilerFlags* flags)
{
	MX_NOTIMPLEMENTED
}

int MxRun_AnyFile(FILE *fp, const char *name)
{
    MX_NOTIMPLEMENTED
}

int MxRun_AnyFileEx(FILE *fp, const char *name, int closeit)
{
    return MxRun_AnyFileExFlags(fp, name, closeit, NULL);
}

int MxRun_AnyFileFlags(FILE *fp, const char *name, MxCompilerFlags *flags)
{
    return MxRun_AnyFileExFlags(fp, name, 0, flags);
}

CObject* MxRun_File(FILE *fp, const char *p, int s, CObject *g, CObject *l)
{
    return MxRun_FileExFlags(fp, p, s, g, l, 0, NULL);
}

CObject* MxRun_FileEx(FILE *fp, const char *p, int s, CObject *g, CObject *l, int c)
{
    return MxRun_FileExFlags(fp, p, s, g, l, c, NULL);
}

CObject* MxRun_FileFlags(FILE *fp, const char *p, int s, CObject *g, CObject *l,
                MxCompilerFlags *flags)
{
    return MxRun_FileExFlags(fp, p, s, g, l, 0, flags);
}

int MxRun_SimpleFile(FILE *f, const char *p)
{
    return MxRun_SimpleFileExFlags(f, p, 0, NULL);
}

int MxRun_SimpleFileEx(FILE *f, const char *p, int c)
{
    return MxRun_SimpleFileExFlags(f, p, c, NULL);
}

CObject* MxRun_String(const char *str, int s, CObject *g, CObject *l)
{
    return MxRun_StringFlags(str, s, g, l, NULL);
}

int MxRun_SimpleString(const char *s)
{
    return MxRun_SimpleStringFlags(s, NULL);
}

int Mx_Main(int argc, const char **argv)
{
	MX_NOTIMPLEMENTED
}






