/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TriangleSetNode.cpp
*
*	Revisions;
*
*	11/27/02
*		- The first revision.
*
******************************************************************/

#include <x3d/TriangleSetNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

TriangleSetNode::TriangleSetNode() 
{
	setHeaderFlag(false);
	setType(TRIANGLESET_NODE);

	///////////////////////////
	// Field 
	///////////////////////////

	// convex  field
	convexField = new SFBool(true);
	convexField->setName(convexFieldString);
	addField(convexField);

	// creaseAngle  field
	creaseAngleField = new SFFloat(0.0f);
	creaseAngleField->setName(creaseAngleFieldString);
	addField(creaseAngleField);
}

TriangleSetNode::~TriangleSetNode() 
{
}
	
////////////////////////////////////////////////
//	Convex
////////////////////////////////////////////////

SFBool *TriangleSetNode::getConvexField() const
{
	if (isInstanceNode() == false)
		return convexField;
	return (SFBool *)getField(convexFieldString);
}
	
void TriangleSetNode::setConvex(bool value) 
{
	getConvexField()->setValue(value);
}

void TriangleSetNode::setConvex(int value)
{
	setConvex(value ? true : false);
}

bool TriangleSetNode::getConvex() const
{
	return getConvexField()->getValue();
}

////////////////////////////////////////////////
//	CreaseAngle
////////////////////////////////////////////////

SFFloat *TriangleSetNode::getCreaseAngleField() const
{
	if (isInstanceNode() == false)
		return creaseAngleField;
	return (SFFloat *)getField(creaseAngleFieldString);
}

void TriangleSetNode::setCreaseAngle(float value) 
{
	getCreaseAngleField()->setValue(value);
}

float TriangleSetNode::getCreaseAngle() const
{
	return getCreaseAngleField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

TriangleSetNode *TriangleSetNode::next() const
{
	return (TriangleSetNode *)Node::next(getType());
}

TriangleSetNode *TriangleSetNode::nextTraversal() const
{
	return (TriangleSetNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool TriangleSetNode::isChildNodeType(Node *node) const
{
	if (node->isColorNode() || node->isCoordinateNode() || node->isNormalNode() || node->isTextureCoordinateNode())
		return true;
	else
		return false;
}

void TriangleSetNode::uninitialize() 
{
}

void TriangleSetNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void TriangleSetNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	TriangleSetNode::initialize
////////////////////////////////////////////////

void TriangleSetNode::initialize() 
{
}

////////////////////////////////////////////////////////////
//	getNPolygons
////////////////////////////////////////////////////////////

int TriangleSetNode::getNPolygons() const
{
	CoordinateNode *coordNode = getCoordinateNodes();
	if (!coordNode)
		return 0;

	int nCoordIndexes = coordNode->getNPoints();
	int nCoordIndex = nCoordIndexes / 3;

	return nCoordIndex;
}

////////////////////////////////////////////////
//	TriangleSetNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void TriangleSetNode::recomputeDisplayList() 
{
}

#endif

