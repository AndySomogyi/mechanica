/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Disk2DNode.cpp
*
******************************************************************/

#include <x3d/Disk2DNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

static const char outerRadiusFieldString[] = "outerRadius";
static const char innerRadiusFieldString[] = "innerRadius";

Disk2DNode::Disk2DNode() 
{
	setHeaderFlag(false);
	setType(DISK2D_NODE);

	///////////////////////////
	// Field 
	///////////////////////////

	// radius field
	outerRadiusField = new SFFloat(1.0f);
	addField(outerRadiusFieldString, outerRadiusField);

	// radius field
	innerRadiusField = new SFFloat(1.0f);
	addField(innerRadiusFieldString, innerRadiusField);
}

Disk2DNode::~Disk2DNode() 
{
}

////////////////////////////////////////////////
//	OuterRadius
////////////////////////////////////////////////

SFFloat *Disk2DNode::getOuterRadiusField() const
{
	if (isInstanceNode() == false)
		return outerRadiusField;
	return (SFFloat *)getField(outerRadiusFieldString);
}
	
void Disk2DNode::setOuterRadius(float value) 
{
	getOuterRadiusField()->setValue(value);
}

float Disk2DNode::getOuterRadius() const 
{
	return getOuterRadiusField()->getValue();
} 

////////////////////////////////////////////////
//	InnerRadius
////////////////////////////////////////////////

SFFloat *Disk2DNode::getInnerRadiusField() const
{
	if (isInstanceNode() == false)
		return innerRadiusField;
	return (SFFloat *)getField(innerRadiusFieldString);
}
	
void Disk2DNode::setInnerRadius(float value) 
{
	getInnerRadiusField()->setValue(value);
}

float Disk2DNode::getInnerRadius() const 
{
	return getInnerRadiusField()->getValue();
} 

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

Disk2DNode *Disk2DNode::next() const 
{
	return (Disk2DNode *)Node::next(getType());
}

Disk2DNode *Disk2DNode::nextTraversal() const 
{
	return (Disk2DNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool Disk2DNode::isChildNodeType(Node *node) const
{
	return false;
}

void Disk2DNode::initialize() 
{
	recomputeBoundingBox();
#ifdef CX3D_SUPPORT_OPENGL
		recomputeDisplayList();
#endif
}

void Disk2DNode::uninitialize()
{
}

void Disk2DNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void Disk2DNode::outputContext(std::ostream &printStream, const char *indentString) const 
{
}

////////////////////////////////////////////////
//	Disk2DNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void Disk2DNode::recomputeDisplayList() 
{
};

#endif
