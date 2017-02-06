/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	X3DParser.cpp
*
******************************************************************/

#include <x3d/CyberX3DConfig.h>
#include <x3d/X3DParser.h>
#include <x3d/X3DParserHandlers.h>
#include <x3d/ParserFunc.h>

#ifdef CX3D_SUPPORT_X3D

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/TransService.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>

using namespace CyberX3D;
using namespace xercesc;

////////////////////////////////////////////////
//	X3DParser
////////////////////////////////////////////////

X3DParser::X3DParser()
{
    try {
         XMLPlatformUtils::Initialize();
    }
	catch (const XMLException& e) {}

	encodingName = "LATIN1";
	expandNamespaces = false;
	//unRepFlags = XMLFormatter::UnRep_CharRef;
	valScheme = SAX2XMLReader::Val_Never;
	doValidation = false;
	doNamespaces = false;
	doSchema = false;
	schemaFullChecking = false;
	namespacePrefixes = false;
}

////////////////////////////////////////////////
//	~X3DParser
////////////////////////////////////////////////

X3DParser::~X3DParser()
{
}

////////////////////////////////////////////////
//	X3DParser::load
////////////////////////////////////////////////

bool X3DParser::load(const char *xmlFile, void (*callbackFn)(int nLine, void *info), void *callbackFnInfo)
{
    SAX2XMLReader* parser = XMLReaderFactory::createXMLReader();

    if (valScheme == SAX2XMLReader::Val_Auto)
    {
        parser->setFeature(XMLUni::fgSAX2CoreValidation, true);
        parser->setFeature(XMLUni::fgXercesDynamic, true);
    }

    if (valScheme == SAX2XMLReader::Val_Never)
    {
        parser->setFeature(XMLUni::fgSAX2CoreValidation, false);
    }

    if (valScheme == SAX2XMLReader::Val_Always)
    {
        parser->setFeature(XMLUni::fgSAX2CoreValidation, true);
        parser->setFeature(XMLUni::fgXercesDynamic, false);
    }

    parser->setFeature(XMLUni::fgSAX2CoreValidation, doValidation); 
    parser->setFeature(XMLUni::fgSAX2CoreNameSpaces, doNamespaces);
    parser->setFeature(XMLUni::fgXercesSchema, doSchema);
    parser->setFeature(XMLUni::fgXercesSchemaFullChecking, schemaFullChecking);
    parser->setFeature(XMLUni::fgSAX2CoreNameSpacePrefixes, namespacePrefixes);

	beginParse();

    int errorCount = 0;
    try
    {
        X3DParserHandlers handler;
        parser->setContentHandler(&handler);
        parser->setErrorHandler(&handler);
        parser->parse(xmlFile);
        errorCount = parser->getErrorCount();
    } catch (const XMLException& e) {
        XMLPlatformUtils::Terminate();
		endParse();
        return false;
    }

    delete parser;

    XMLPlatformUtils::Terminate();
	endParse();

	if (0 < errorCount)
		return false;

	return true;
}

#endif
