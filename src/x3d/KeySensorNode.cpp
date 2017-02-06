/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	KeySensorNode.cpp
*
******************************************************************/

#include <x3d/KeySensorNode.h>

using namespace CyberX3D;

KeySensorNode::KeySensorNode() 
{
	setHeaderFlag(false);
	setType(KEYSENSOR_NODE);
}

KeySensorNode::~KeySensorNode() 
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

KeySensorNode *KeySensorNode::next() const
{
	return (KeySensorNode *)Node::next(getType());
}

KeySensorNode *KeySensorNode::nextTraversal() const
{
	return (KeySensorNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool KeySensorNode::isChildNodeType(Node *node) const
{
	return false;
}

void KeySensorNode::initialize() 
{
}

void KeySensorNode::uninitialize() 
{
}

void KeySensorNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void KeySensorNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}
