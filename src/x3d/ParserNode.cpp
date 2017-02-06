/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ParserNode.cpp
*
******************************************************************/

#include <x3d/ParserNode.h>

using namespace CyberX3D;

ParserNode::ParserNode(Node *node, int type) 
{ 
	setHeaderFlag(false); 
	mNode = node; 
	mType = type;
}

ParserNode::~ParserNode() 
{ 
	remove();
}
	
Node *ParserNode::getNode() const
{ 
	return mNode; 
}
	
int ParserNode::getType() const
{ 
	return mType; 
}
