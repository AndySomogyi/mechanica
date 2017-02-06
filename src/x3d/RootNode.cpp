/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	RouteNode.cpp
*
******************************************************************/

#include <x3d/RootNode.h>

using namespace CyberX3D;

RootNode::RootNode() 
{
	setHeaderFlag(true);
	setType(ROOT_NODE);
}

RootNode::~RootNode() 
{
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool RootNode::isChildNodeType(Node *node) const
{
	if (node->isCommonNode() || node->isBindableNode() ||node->isInterpolatorNode() || node->isSensorNode() || node->isGroupingNode() || node->isSpecialGroupNode())
		return true;
	else
		return false;
}

void RootNode::initialize() 
{
}

void RootNode::uninitialize() 
{
}

void RootNode::update() 
{
}

////////////////////////////////////////////////
//	infomation
////////////////////////////////////////////////

void RootNode::outputContext(std::ostream& printStream, const char * indentString) const
{
}
