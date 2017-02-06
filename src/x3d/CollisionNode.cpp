/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	CollisionNode.cpp
*
*	Revisions:
*
*	11/19/02
*		- Changed the super class from Node to BoundedGroupingNode.
*
******************************************************************/

#include <x3d/VRML97Fields.h>
#include <x3d/CollisionNode.h>

using namespace CyberX3D;

CollisionNode::CollisionNode() 
{
	setHeaderFlag(false);
	setType(COLLISION_NODE);

	// collide exposed field
	collideField = new SFBool(true);
	addExposedField(collideFieldString, collideField);

	// collide event out
	collideTimeField = new SFTime(-1.0);
	addEventOut(collideTimeFieldString, collideTimeField);
}

CollisionNode::~CollisionNode() 
{
}

////////////////////////////////////////////////
//	collide
////////////////////////////////////////////////

SFBool *CollisionNode::getCollideField() const
{
	if (isInstanceNode() == false)
		return collideField;
	return (SFBool *)getExposedField(collideFieldString);
}

void CollisionNode::setCollide(bool  value) 
{
	getCollideField()->setValue(value);
}

void CollisionNode::setCollide(int value) 
{
	setCollide(value ? true : false);
}

bool CollisionNode::getCollide() const 
{
	return getCollideField()->getValue();
}

////////////////////////////////////////////////
//	collideTime
////////////////////////////////////////////////

SFTime *CollisionNode::getCollideTimeField() const
{
	if (isInstanceNode() == false)
		return collideTimeField;
	return (SFTime *)getEventOut(collideTimeFieldString);
}

void CollisionNode::setCollideTime(double value) 
{
	getCollideTimeField()->setValue(value);
}

double CollisionNode::getCollideTime() const 
{
	return getCollideTimeField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

CollisionNode *CollisionNode::next() const 
{
	return (CollisionNode *)Node::next(getType());
}

CollisionNode *CollisionNode::nextTraversal() const 
{
	return (CollisionNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool CollisionNode::isChildNodeType(Node *node) const
{
	if (node->isCommonNode() || node->isBindableNode() ||node->isInterpolatorNode() || node->isSensorNode() || node->isGroupingNode() || node->isSpecialGroupNode())
		return true;
	else
		return false;
}

void CollisionNode::initialize() 
{
	recomputeBoundingBox();
}

void CollisionNode::uninitialize() 
{
}

void CollisionNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void CollisionNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFBool *collide = getCollideField();
	printStream << indentString << "\t" << "collide " << collide << std::endl;
}
