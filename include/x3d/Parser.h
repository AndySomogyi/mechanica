/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Parser.h
*
******************************************************************/

#ifndef _CX3D_PARSER_H_
#define _CX3D_PARSER_H_

#include <x3d/CyberX3DConfig.h>
#include <x3d/Scene.h>
#include <x3d/ParserNode.h>
#include <x3d/ParserResult.h>
#include <x3d/PROTO.h>
#include <x3d/DEF.h>
#include <x3d/StringUtil.h>
#include <x3d/LinkedListNode.h>

namespace CyberX3D {

const int DEFAULT_LEX_LINE_BUFFER_SIZE = 64*1024;

class Parser : public Scene, public ParserResult, public LinkedListNode<Parser> {

	LinkedList<ParserNode>	mParserNodeList;
	LinkedList<DEF>		mDEFList;
	LinkedList<PROTO>		mPROTOList;
	String					mDefName;

	bool					mbParsering;

public:

	Parser();
	virtual ~Parser();

	///////////////////////////////////////////////
	//	Praser action
	///////////////////////////////////////////////

	void addNode(Node *node, bool initialize = true);
	void addNodeAtFirst(Node *node, bool initialize = true);

	void moveNode(Node *node);
	void moveNodeAtFirst(Node *node);

	void pushNode(Node *node, int type);
	void popNode();

	Node *getCurrentNode();
	int getCurrentNodeType() const;
	int getPrevNodeType() const;

	void beginParse();
	void endParse();

	///////////////////////////////////////////////
	//	DEF
	///////////////////////////////////////////////

	void setDefName(const char *name) {
		mDefName.setValue(name);
	}

	const char *getDefName() const {
		return mDefName.getValue();
	}

	///////////////////////////////////////////////
	//	Load
	///////////////////////////////////////////////

	void	setParseringState(bool state)	{ mbParsering = state; }
	bool	getParseringState()				{ return mbParsering; }

	int		getNLines(const char *fileName) const;

	virtual bool load(const char *fileName, void (*callbackFn)(int nLine, void *info) = NULL, void *callbackFnInfo = NULL) = 0;

	///////////////////////////////////////////////
	//	DEF
	///////////////////////////////////////////////

	DEF *getDEFs() const;
	const char *getDEFString(const char *name) const;
	void addDEF(DEF *def);
	void addDEF(const char *name, const char *string);
	void deleteDEFs();

	///////////////////////////////////////////////
	//	PROTO
	///////////////////////////////////////////////

	PROTO *getPROTOs() const;
	PROTO *getPROTO(const char *name) const;
	void addPROTO(PROTO *proto);
	void deletePROTOs();
};

}

#endif
