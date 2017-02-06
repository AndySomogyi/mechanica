/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	RouteNode.cpp
*
******************************************************************/

#include <x3d/RouteNode.h>

using namespace CyberX3D;

static const char fromNodeString[] = "fromNode";
static const char fromFieldString[] = "fromField";
static const char toNodeString[] = "toNode";
static const char toFieldString[] = "toField";

RouteNode::RouteNode() 
{
	// fromNode
	fromNode = new SFString();
	addExposedField(fromNodeString, fromNode);

	// fromField
	fromField = new SFString();
	addExposedField(fromFieldString, fromField);

	// toNode
	toNode = new SFString();
	addExposedField(toNodeString, toNode);

	// toField
	toField = new SFString();
	addExposedField(toFieldString, toField);
}

RouteNode::~RouteNode() 
{
}

////////////////////////////////////////////////
//	FromField
////////////////////////////////////////////////

SFString *RouteNode::getFromField() const
{
	if (isInstanceNode() == false)
		return fromField;
	return (SFString *)getExposedField(fromFieldString);
}
	
void RouteNode::setFromFieldName(const char *value) 
{
	getFromField()->setValue(value);
}

const char *RouteNode::getFromFieldName() const
{
	return getFromField()->getValue();
}

////////////////////////////////////////////////
//	FromField
////////////////////////////////////////////////

SFString *RouteNode::getFromNode() const
{
	if (isInstanceNode() == false)
		return fromNode;
	return (SFString *)getExposedField(fromNodeString);
}
	
void RouteNode::setFromNodeName(const char *value) 
{
	getFromNode()->setValue(value);
}

const char *RouteNode::getFromNodeName() const
{
	return getFromNode()->getValue();
}

////////////////////////////////////////////////
//	ToField
////////////////////////////////////////////////

SFString *RouteNode::getToField() const
{
	if (isInstanceNode() == false)
		return toField;
	return (SFString *)getExposedField(toFieldString);
}
	
void RouteNode::setToFieldName(const char *value) 
{
	getToField()->setValue(value);
}

const char *RouteNode::getToFieldName() const
{
	return getToField()->getValue();
}

////////////////////////////////////////////////
//	ToNode
////////////////////////////////////////////////

SFString *RouteNode::getToNode() const
{
	if (isInstanceNode() == false)
		return toNode;
	return (SFString *)getExposedField(toNodeString);
}
	
void RouteNode::setToNodeName(const char *value) 
{
	getToNode()->setValue(value);
}

const char *RouteNode::getToNodeName() const
{
	return getToNode()->getValue();
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void RouteNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool RouteNode::isChildNodeType(Node *node) const
{
	return false;
}

void RouteNode::initialize() 
{
}

void RouteNode::uninitialize() 
{
}

void RouteNode::update() 
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

RouteNode *RouteNode::next() const
{
	return (RouteNode *)Node::next(getType());
}

RouteNode *RouteNode::nextTraversal() const
{
	return (RouteNode *)Node::nextTraversalByType(getType());
}

