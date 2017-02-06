/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 2002
*
*	File:	X3DParserHandlers.cpp
*
******************************************************************/

#include <string>
#include <x3d/X3DParser.h>
#include <x3d/X3DParserHandlers.h>
#include <x3d/X3DParserTokenizer.h>
#include <x3d/ParserFunc.h>
#include <x3d/XMLNode.h>

#ifdef CX3D_SUPPORT_X3D

#include <xercesc/util/XMLUniDefs.hpp>
#include <xercesc/sax2/Attributes.hpp>

using namespace std;
using namespace CyberX3D;
using namespace xercesc;

////////////////////////////////////////////////////////////////////////////////////////////////
//	XMLCh Utils
////////////////////////////////////////////////////////////////////////////////////////////////

char *XMLCh2Char(const XMLCh *value, char *transBuf, int bufSize)
{
	if (XMLString::transcode(value, transBuf, bufSize-1) == true)
		return transBuf;
	return "";
}

char *XMLCh2Char(const XMLCh *value)
{
	static const int TRANS_BUF_SIZE = 1024;
	static char transBuf[TRANS_BUF_SIZE];
	return XMLCh2Char(value, transBuf, TRANS_BUF_SIZE);
}

char *XMLCh2NewChar(const XMLCh *value)
{
	return XMLString::transcode(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//  X3DParserHandlers: Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////////////////////

X3DParserHandlers::X3DParserHandlers()
{
}

X3DParserHandlers::~X3DParserHandlers()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////
//  X3DParserHandlers: Overrides of the output formatter target interface
////////////////////////////////////////////////////////////////////////////////////////////////

void X3DParserHandlers::writeChars(const XMLByte* const toWrite) const
{
}

void X3DParserHandlers::writeChars(
const XMLByte* const toWrite,
const unsigned int count,
XMLFormatter* const formatter) const
{
}

////////////////////////////////////////////////////////////////////////////////////////////////
//  X3DParserHandlers: Overrides of the SAX ErrorHandler interface
////////////////////////////////////////////////////////////////////////////////////////////////

void X3DParserHandlers::exception(const SAXParseException& e)
{
	ParserResult *presult = GetParserResultObject();
	presult->setErrorLineNumber(e.getLineNumber());
	presult->setErrorToken(XMLCh2Char(e.getPublicId()));
	presult->setErrorMessage(XMLCh2Char(e.getMessage()));
	presult->setErrorLineString(XMLCh2Char(e.getPublicId()));
}

void X3DParserHandlers::error(const SAXParseException& e)
{
	exception(e);
}

void X3DParserHandlers::fatalError(const SAXParseException& e)
{
	exception(e);
}

void X3DParserHandlers::warning(const SAXParseException& e)
{
	exception(e);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//  X3DParserHandlers: Overrides of the SAX DTDHandler interface
////////////////////////////////////////////////////////////////////////////////////////////////

void X3DParserHandlers::unparsedEntityDecl(
const XMLCh* const name,
const XMLCh* const publicId,
const XMLCh* const systemId,
const XMLCh* const notationName)
{
    // Not used at this time
}


void X3DParserHandlers::notationDecl(
const   XMLCh* const name,
const XMLCh* const publicId,
const XMLCh* const systemId)
{
    // Not used at this time
}

////////////////////////////////////////////////////////////////////////////////////////////////
//  X3DParserHandlers: Overrides of the SAX DocumentHandler interface
////////////////////////////////////////////////////////////////////////////////////////////////

void X3DParserHandlers::characters(
const XMLCh* const chars,
const unsigned int length)
{
}

void X3DParserHandlers::ignorableWhitespace(
const XMLCh* const chars,
const  unsigned int length)
{
}

void X3DParserHandlers::processingInstruction(
const  XMLCh* const target, 
const XMLCh* const data)
{
}

void X3DParserHandlers::startDocument()
{
}

void X3DParserHandlers::endDocument()
{
}

void X3DParserHandlers::addXMLElement(XMLNode *xmlNode,  const char *attrName, const XMLCh *attrValue)
{
	char *attrValueStr = XMLCh2Char(attrValue);
	xmlNode->addElement(attrName, attrValueStr);
}

void X3DParserHandlers::addX3DElement(Node *x3dNode, const char *attrName, const XMLCh *attrValue)
{
	Field *field = x3dNode->findField(attrName);
	if (field == NULL)
		return;

	if (field->isSField() == true) {
		char *attrValueStr = XMLCh2Char(attrValue);
		field->setValue(attrValueStr);
		return;
	}

	MField *mfield = (MField *)field;
	int mfieldCnt = mfield->getValueCount();

	char *attrValueStr = XMLCh2Char(attrValue);
	X3DParserTokenizer attrToken(attrValue);
	int tokenCnt = 0;
	string fieldTokenStr;
	while (attrToken.hasMoreTokens() == true) { 
		tokenCnt++;
		XMLCh *token = attrToken.nextToken();
		char *tokenValue = XMLCh2Char(token);
		if (0 < fieldTokenStr.length())
			fieldTokenStr.append(" ");
		fieldTokenStr.append(tokenValue);
		if (mfieldCnt <= tokenCnt) {
			mfield->addValue(fieldTokenStr.c_str());
			tokenCnt = 0;
			fieldTokenStr = "";
		}
	}
}

void X3DParserHandlers::startElement(
const XMLCh* const uri,
const XMLCh* const localname,
const XMLCh* const qname,
const Attributes& attributes)
{
	char* elemName = XMLString::transcode(qname);
	
	int nodeType = GetNodeType(elemName);
	Node *node = CreateX3DNode(nodeType);
	
	if (node->isXMLNode() == true)
		node->setName(elemName);

    unsigned int len = attributes.getLength();
    for (unsigned int index = 0; index < len; index++)
    {
		std::string attrName = XMLCh2Char(attributes.getQName(index));
		const XMLCh *attrValue = attributes.getValue(index);
		if (node->isXMLNode() == true) {
			XMLNode *xmlNode = (XMLNode *)node;
			addXMLElement(xmlNode, attrName.c_str(), attrValue);
		}
		else 
			addX3DElement(node, attrName.c_str(), attrValue);
    }

	ParserAddNode(node);
	ParserPushNode(node);
}

void X3DParserHandlers::endElement(
const XMLCh* const uri,
const XMLCh* const localname,
const XMLCh* const qname)
{
	char* elemName = XMLString::transcode(qname);
	ParserPopNode();
}

#endif
