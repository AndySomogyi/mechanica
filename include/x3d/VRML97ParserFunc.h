/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	VRML97ParserFunc.h
*
******************************************************************/

#ifndef _CX3D_VRML97PARSERFUNC_H_
#define _CX3D_VRML97PARSERFUNC_H_

#include <stdio.h>
#include <x3d/ParserFunc.h>

namespace CyberX3D {

class PROTO;

////////////////////////////////////////////////
// AddSF*
////////////////////////////////////////////////

void AddSFColor(float color[3]);
void AddSFRotation(float rotation[4]);
void AddSFVec3f(float vector[3]);
void AddSFVec2f(float vector[2]);
void AddSFInt32(int value);
void AddSFFloat(float value);
void AddSFString(const char *string);

////////////////////////////////////////////////
// PROTO/DEF
////////////////////////////////////////////////

PROTO *AddPROTOInfo(const char *name, const char *string, const char *fieldString);
PROTO *IsPROTOName(const char *name);

void AddDEFInfo(const char *name, const char *string);
const char *GetDEFSrting(const char *name);

void SetDEFName(const char *name);
const char *GetDEFName(void);

////////////////////////////////////////////////
// lex
////////////////////////////////////////////////
void MakeLexerStringBuffers(int lexBufferSize, int lineBufferSize);
void DeleteLexerStringBuffers(void);

void MakeLexerBuffers(int lexBufferSize, int lineBufferSize);
void DeleteLexerBuffers(void);
void SetLexCallbackFn(void (*func)(int nLine, void *info), void *fnInfo);
int UnputString(const char *pBegin);
void CurrentLineIncrement();

////////////////////////////////////////////////
// yacc
////////////////////////////////////////////////

int GetCurrentLineNumber(void);
void SetInputFile(FILE *fp);
void SetInputBuffer(const char *, int);
const char *GetErrorLineString(void);

}

/******************************************************************
*	yacc
******************************************************************/

int yyparse(void);

#endif
