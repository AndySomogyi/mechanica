/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NormalNode.cpp
*
*	Revisions:
*
*	11/25/02
*		- Changed the super class from Node to GeometricPropertyNode.
*
******************************************************************/

#include <x3d/NormalNode.h>

using namespace CyberX3D;

NormalNode::NormalNode () 
{
	setHeaderFlag(false);
	setType(NORMAL_NODE);

	// vector exposed field
	vectorField = new MFVec3f();
	vectorField->setName(vectorFieldString);
	addExposedField(vectorField);
}

NormalNode::~NormalNode () 
{
}

////////////////////////////////////////////////
//	vector
////////////////////////////////////////////////

MFVec3f *NormalNode::getVectorField() const
{
	if (isInstanceNode() == false)
		return vectorField;
	return (MFVec3f *)getExposedField(vectorFieldString);
}
	
void NormalNode::addVector(float value[]) 
{
	getVectorField()->addValue(value);
}

int NormalNode::getNVectors() const
{
	return getVectorField()->getSize();
}

void NormalNode::getVector(int index, float value[]) const
{
	getVectorField()->get1Value(index, value);
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool NormalNode::isChildNodeType(Node *node) const
{
	return false;
}

void NormalNode::initialize() 
{
}

void NormalNode::uninitialize() 
{
}

void NormalNode::update() 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void NormalNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	if (0 < getNVectors()) {
		MFVec3f *vector = getVectorField();
		printStream <<  indentString << "\t" << "vector ["  << std::endl;
		vector->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

NormalNode *NormalNode::next() const
{
	return (NormalNode *)Node::next(getType());
}

NormalNode *NormalNode::nextTraversal() const
{
	return (NormalNode *)Node::nextTraversalByType(getType());
}

