/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ParserNode.h
*
******************************************************************/

#ifndef _CX3D_PARSERNODE_H_
#define _CX3D_PARSERNODE_H_

#include <x3d/LinkedList.h>
#include <x3d/Node.h>

namespace CyberX3D {

class ParserNode : public LinkedListNode<ParserNode> {
	Node		*mNode;
	int			mType;
public:

	ParserNode(Node *node, int type);
	virtual ~ParserNode();
	
	Node *getNode() const; 
	int getType() const;
};

}

#endif

