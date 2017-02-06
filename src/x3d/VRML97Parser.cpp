/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	VRML97Parser.cpp
*
*      03/12/04
*              - Added the following functions to parse more minimum memory.
*                VRML97ParserSetBufSize(), VRML97ParserGetBufSize().
*
******************************************************************/

#include <stdio.h>
#include <x3d/VRML97Parser.h>
#include <x3d/VRML97ParserFunc.h>
#if defined(CX3D_SUPPORT_GZIP)
#include <zlib.h>
#endif

using namespace CyberX3D;

////////////////////////////////////////////////
//	VRML97Parser::load
////////////////////////////////////////////////

VRML97Parser::VRML97Parser() 
{
	setParserResult(false);
	setParseringState(false);
}

VRML97Parser::~VRML97Parser() 
{
}

////////////////////////////////////////////////
//	VRML97Parser::load
////////////////////////////////////////////////

bool VRML97Parser::load(const char *fileName, void (*callbackFn)(int nLine, void *info), void *callbackFnInfo) 
{
	FILE *fp = fopen(fileName, "rb");

#if defined(CX3D_SUPPORT_URL)
	SceneGraph *sg = (SceneGraph *)this;
#endif

#if defined(CX3D_SUPPORT_URL)
	if (fp == NULL){
		if (sg->getUrlStream(fileName)) {
			char *outputFilename = sg->getUrlOutputFilename();
			fp = fopen(outputFilename, "rb");
			sg->setUrl(fileName);
		}
	}
#endif

	if (fp == NULL) {
		fprintf(stderr, "Cannot open data file %s\n", fileName);
		setParserResult(false);
		setParseringState(false);
		return false;
	}

#if defined(CX3D_SUPPORT_GZIP)
	gzFile gfd = gzdopen(fileno(fp), "rb");

	if (gfd == 0){
		return false;
	}
	std::string buffer = "";
    char buf[64];
	while (!gzeof(gfd)) {
          memset(buf, '\0', 64);
	  int size = gzread(gfd, &buf[0], 63);
	  buffer += buf;
	}
    int lexBufferSize = buffer.size();	
    SetInputBuffer(buffer.c_str(), buffer.size());
#else
	#ifdef USE_MAX_LEX_BUFFER
	fseek(fp, 0, SEEK_END);
	int lexBufferSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	#else
	int lexBufferSize = VRML97ParserGetBufSize();
	#endif
#endif

	if (GetParserObject() == NULL) {
#if defined(CX3D_SUPPORT_GZIP)
		MakeLexerStringBuffers(lexBufferSize, DEFAULT_LEX_LINE_BUFFER_SIZE);
#else
		MakeLexerBuffers(lexBufferSize, DEFAULT_LEX_LINE_BUFFER_SIZE);
#endif
	}

	clearNodeList();
	clearRouteList();
	deleteDEFs();
	deletePROTOs();

	beginParse();

	SetLexCallbackFn(callbackFn, callbackFnInfo);
#if !defined(CX3D_SUPPORT_GZIP)
	SetInputFile(fp);
#endif
    setParserResult(!yyparse() ? true : false);

	endParse();

	if (GetParserObject() == NULL) {
#if defined(CX3D_SUPPORT_GZIP)
		DeleteLexerStringBuffers();
#else
		DeleteLexerBuffers();
#endif
	}

#if defined(CX3D_SUPPORT_GZIP)
   gzclose(gfd);
#endif
	fclose(fp);

#if defined(CX3D_SUPPORT_URL)
	sg->deleteUrlOutputFilename();
#endif

	return getParserResult();
}

////////////////////////////////////////////////
//     VRML97Parser::load
////////////////////////////////////////////////

static int vrml97parserBufSize = VRML97_PARSER_DEFAULT_BUF_SIZE;

void  CyberX3D::VRML97ParserSetBufSize(int bufSize)
{
       vrml97parserBufSize = bufSize;
}

int CyberX3D::VRML97ParserGetBufSize()
{
       return vrml97parserBufSize;
}

