/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	DEFNode.cpp
*
******************************************************************/

#include <x3d/DEFNode.h>

using namespace CyberX3D;

DEFNode::DEFNode() 
{
	setHeaderFlag(false);
	setType(DEF_NODE);
}

DEFNode::~DEFNode() 
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

DEFNode *DEFNode::next() const 
{
	return (DEFNode *)Node::next(getType());
}

DEFNode *DEFNode::nextTraversal() const 
{
	return (DEFNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool DEFNode::isChildNodeType(Node *node) const
{
	return false;
}

void DEFNode::initialize()
{
}

void DEFNode::uninitialize()
{
}

void DEFNode::update() 
{
}

void DEFNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

