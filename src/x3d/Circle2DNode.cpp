/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Circle2DNode.cpp
*
******************************************************************/

#include <x3d/Circle2DNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

Circle2DNode::Circle2DNode() 
{
	setHeaderFlag(false);
	setType(CIRCLE2D_NODE);

	///////////////////////////
	// Field 
	///////////////////////////

	// radius field
	radiusField = new SFFloat(1.0f);
	addField(radiusFieldString, radiusField);
}

Circle2DNode::~Circle2DNode() 
{
}

////////////////////////////////////////////////
//	Radius
////////////////////////////////////////////////

SFFloat *Circle2DNode::getRadiusField() const
{
	if (isInstanceNode() == false)
		return radiusField;
	return (SFFloat *)getField(radiusFieldString);
}
	
void Circle2DNode::setRadius(float value) 
{
	getRadiusField()->setValue(value);
}

float Circle2DNode::getRadius() const 
{
	return getRadiusField()->getValue();
} 

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

Circle2DNode *Circle2DNode::next() const 
{
	return (Circle2DNode *)Node::next(getType());
}

Circle2DNode *Circle2DNode::nextTraversal() const 
{
	return (Circle2DNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool Circle2DNode::isChildNodeType(Node *node) const
{
	return false;
}

void Circle2DNode::initialize() 
{
	recomputeBoundingBox();
#ifdef CX3D_SUPPORT_OPENGL
		recomputeDisplayList();
#endif
}

void Circle2DNode::uninitialize() 
{
}

void Circle2DNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void Circle2DNode::outputContext(std::ostream &printStream, const char *indentString) const 
{
}

////////////////////////////////////////////////
//	Circle2DNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void Circle2DNode::recomputeDisplayList() 
{
};

#endif
