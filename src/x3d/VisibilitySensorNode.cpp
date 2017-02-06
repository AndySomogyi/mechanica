/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	VisibilitySensorNode.cpp
*
*	Revisions:
*
*	12/08/02
*		- Changed the super class from SensorNode to EnvironmentalSensorNode.
*		- Moved the following fields to EnvironmentalSensorNode.
*			center, size, enterTime, exitTime
*
******************************************************************/

#include <x3d/VisibilitySensorNode.h>

using namespace CyberX3D;

VisibilitySensorNode::VisibilitySensorNode() 
{
	setHeaderFlag(false);
	setType(VISIBILITYSENSOR_NODE);
}

VisibilitySensorNode::~VisibilitySensorNode() 
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

VisibilitySensorNode *VisibilitySensorNode::next() const
{
	return (VisibilitySensorNode *)Node::next(getType());
}

VisibilitySensorNode *VisibilitySensorNode::nextTraversal() const
{
	return (VisibilitySensorNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool VisibilitySensorNode::isChildNodeType(Node *node) const
{
	return false;
}

void VisibilitySensorNode::initialize() 
{
}

void VisibilitySensorNode::uninitialize() 
{
}

void VisibilitySensorNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void VisibilitySensorNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFBool *enabled = getEnabledField();
	SFVec3f *center = getCenterField();
	SFVec3f *size = getSizeField();
	printStream << indentString << "\t" << "enabled " << enabled << std::endl;
	printStream << indentString << "\t" << "center " << center << std::endl;
	printStream << indentString << "\t" << "size " << size << std::endl;
}


