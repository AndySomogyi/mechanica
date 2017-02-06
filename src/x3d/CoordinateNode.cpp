/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	CoordinateNode.cpp
*
*	Revisions:
*
*	11/25/02
*		- Changed the super class from Node to GeometricPropertyNode.
*
******************************************************************/

#include <x3d/CoordinateNode.h>

using namespace CyberX3D;

CoordinateNode::CoordinateNode() 
{
	setHeaderFlag(false);
	setType(COORDINATE_NODE);

	// point exposed field
	pointField = new MFVec3f();
	pointField->setName(pointFieldString);
	addExposedField(pointField);
}

CoordinateNode::~CoordinateNode() {
}

////////////////////////////////////////////////
//	point 
////////////////////////////////////////////////

MFVec3f *CoordinateNode::getPointField() const
{
	if (isInstanceNode() == false)
		return pointField;
	return 	(MFVec3f *)getExposedField(pointFieldString);
}

void CoordinateNode::addPoint(float point[]) 
{
	getPointField()->addValue(point);
}

void CoordinateNode::addPoint(float x, float y, float z) 
{
	getPointField()->addValue(x, y, z);
}

int CoordinateNode::getNPoints()  const
{
	return getPointField()->getSize();
}

void CoordinateNode::getPoint(int index, float point[])  const
{
	getPointField()->get1Value(index, point);
}

void CoordinateNode::setPoint(int index, float point[]) 
{
	getPointField()->set1Value(index, point);
}

void CoordinateNode::setPoint(int index, float x, float y, float z) 
{
	getPointField()->set1Value(index, x, y, z);
}

void CoordinateNode::removePoint(int index) 
{
	getPointField()->remove(index);
}

void CoordinateNode::removeLastPoint() 
{
	getPointField()->removeLastObject();
}

void CoordinateNode::removeFirstPoint() 
{
	getPointField()->removeFirstObject();
}

void CoordinateNode::removeAllPoints() 
{
	getPointField()->clear();
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool CoordinateNode::isChildNodeType(Node *node) const
{
	return false;
}

void CoordinateNode::initialize() 
{
}

void CoordinateNode::uninitialize() 
{
}

void CoordinateNode::update() 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void CoordinateNode::outputContext(std::ostream &printStream, const char *indentString) const 
{
	if (0 < getNPoints()) {
		MFVec3f *point = getPointField();
		printStream <<  indentString << "\t" << "point ["  << std::endl;
		point->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

CoordinateNode *CoordinateNode::next()  const
{
	return (CoordinateNode *)Node::next(getType());
}

CoordinateNode *CoordinateNode::nextTraversal()  const
{
	return (CoordinateNode *)Node::nextTraversalByType(getType());
}
