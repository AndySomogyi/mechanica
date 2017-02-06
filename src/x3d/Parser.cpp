/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Parser.cpp
*
******************************************************************/

#include <stdlib.h>
#include <x3d/Parser.h>
#include <x3d/UrlFile.h>
#include <x3d/ParserFunc.h>

using namespace CyberX3D;

////////////////////////////////////////////////
//	Parser
////////////////////////////////////////////////

Parser::Parser() 
{
	setParserResult(false);
	setParseringState(false);
}

Parser::~Parser() 
{
}

////////////////////////////////////////////////
//	Parser::getNLines
////////////////////////////////////////////////

int Parser::getNLines(const char *fileName) const
{
	FILE *fp;
	if ((fp = fopen(fileName, "rt")) == NULL){
		fprintf(stderr, "Cannot open data file %s\n", fileName);
		return 0;
	}

	char *lineBuffer = (char *)malloc(DEFAULT_LEX_LINE_BUFFER_SIZE + 1);

	int nLine = 0;
	while (fgets(lineBuffer, DEFAULT_LEX_LINE_BUFFER_SIZE, fp))
		nLine++;

	delete lineBuffer;

	fclose(fp);

	return nLine;
}

////////////////////////////////////////////////
//	Parse Action
////////////////////////////////////////////////

void Parser::addNode(Node *node, bool initialize) 
{
	moveNode(node);
	if (initialize)
		node->initialize();
}

void Parser::addNodeAtFirst(Node *node, bool initialize) 
{
	moveNodeAtFirst(node);
	if (initialize)
		node->initialize();
}

void Parser::moveNode(Node *node) 
{
	Node *parentNode = getCurrentNode();
	if (!parentNode || !getParseringState())
		getNodeList()->addNode(node);
	else
		parentNode->moveChildNode(node);

	node->setParentNode(parentNode);
	//node->setSceneGraph((SceneGraph *)this);
}

void Parser::moveNodeAtFirst(Node *node) 
{
	Node *parentNode = getCurrentNode();
	if (!parentNode || !getParseringState())
		getNodeList()->addNodeAtFirst(node);
	else
		parentNode->moveChildNodeAtFirst(node);

	node->setParentNode(parentNode);
	//node->setSceneGraph((SceneGraph *)this);
}

void Parser::pushNode(Node *node, int type)
{
	ParserNode *parserNode = new ParserNode(node, type);
	mParserNodeList.addNode(parserNode);
}

void Parser::popNode()
{
	ParserNode *lastNode = mParserNodeList.getLastNode(); 
	delete lastNode;
}

Node *Parser::getCurrentNode()
{
	ParserNode *lastNode = mParserNodeList.getLastNode(); 
	if (!lastNode)
		return NULL;
	else
		return lastNode->getNode();
}

int Parser::getCurrentNodeType() const
{
	ParserNode *lastNode = mParserNodeList.getLastNode(); 
	if (!lastNode)
		return 0;
	else
		return lastNode->getType();
}

int Parser::getPrevNodeType() const
{
	ParserNode *lastNode = mParserNodeList.getLastNode(); 
	if (!lastNode)
		return 0;
	else {
		ParserNode *prevNode = lastNode->prev(); 
		if (prevNode->isHeaderNode())
			return 0;
		else
			return prevNode->getType();
	}
}

///////////////////////////////////////////////
//	begin/endParse
///////////////////////////////////////////////

void Parser::beginParse()
{
	PushParserObject(this);
	setParseringState(true);
}

void Parser::endParse()
{
	PopParserObject();
	setParseringState(false);
}

///////////////////////////////////////////////
//	DEF
///////////////////////////////////////////////

DEF *Parser::getDEFs() const
{
	return (DEF *)mDEFList.getNodes();
}

const char *Parser::getDEFString(const char *name) const
{
	for (DEF *def=getDEFs(); def; def=def->next()) {
		const char *defName = def->getName();
		if (defName && !strcmp(defName, name))
			return def->getString();
	}
	return NULL;
}

void Parser::addDEF(DEF *def) 
{
	mDEFList.addNode(def);
}
	
void Parser::addDEF(const char *name, const char *string) 
{
	DEF *def = new DEF(name, string);
	addDEF(def);
}

void Parser::deleteDEFs() 
{
	mDEFList.deleteNodes();
}

///////////////////////////////////////////////
//	PROTO
///////////////////////////////////////////////

PROTO *Parser::getPROTOs() const
{
	return (PROTO *)mPROTOList.getNodes();
}

PROTO *Parser::getPROTO(const char *name) const
{
	if (!name || !strlen(name))
		return NULL;

	for (PROTO *proto=getPROTOs(); proto; proto=proto->next()) {
		const char *protoName = proto->getName();
		if (protoName && !strcmp(protoName, name))
			return proto;
	}
	return NULL;
}

void Parser::addPROTO(PROTO *proto) 
{
	mPROTOList.addNode(proto);
}

void Parser::deletePROTOs() 
{
	mPROTOList.deleteNodes();
}

