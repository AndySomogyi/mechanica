/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	X3DParserTokenizer.cpp
*
******************************************************************/

#include <x3d/X3DParserTokenizer.h>

#ifdef CX3D_SUPPORT_X3D

using namespace CyberX3D;
using namespace xercesc;

////////////////////////////////////////////////
//	X3DParserTokenizer
////////////////////////////////////////////////

static const char *tokenDelim = " ,\"\t\r\n";

X3DParserTokenizer::X3DParserTokenizer(const XMLCh *const srcStr) : XMLStringTokenizer(srcStr, XMLString::transcode(tokenDelim))
{
}

////////////////////////////////////////////////
//	~X3DParserTokenizer
////////////////////////////////////////////////

X3DParserTokenizer::~X3DParserTokenizer()
{
}

#endif

