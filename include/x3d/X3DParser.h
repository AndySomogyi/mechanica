/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	X3DParser.h
*
******************************************************************/

#ifndef _CX3D_X3DPARSER_H_
#define _CX3D_X3DPARSER_H_

#ifdef CX3D_SUPPORT_X3D

#include <xercesc/framework/XMLFormatter.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>

#include <x3d/Parser.h>

namespace CyberX3D {

class X3DParser : public Parser {

public:

	xercesc::XMLFormatter::UnRepFlags unRepFlags;
	xercesc::SAX2XMLReader::ValSchemes valScheme;

	char *encodingName;
	bool expandNamespaces;
	bool doNamespaces;
	bool doValidation;
	bool doSchema;
	bool schemaFullChecking;
	bool namespacePrefixes;

public:

	X3DParser();
	virtual ~X3DParser();

	bool load(const char *fileName, void (*callbackFn)(int nLine, void *info) = NULL, void *callbackFnInfo = NULL);
};

}

#endif

#endif
