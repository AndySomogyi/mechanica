/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	CoordinateInterpolator2DNode.cpp
*
******************************************************************/

#include <x3d/CoordinateInterpolator2DNode.h>

using namespace CyberX3D;

CoordinateInterpolator2DNode::CoordinateInterpolator2DNode() 
{
	setHeaderFlag(false);
	setType(COORDINATEINTERPOLATOR2D_NODE);

	// keyValue exposed field
	keyValueField = new MFVec2f();
	addExposedField(keyValueFieldString, keyValueField);

	// value_changed eventOut field
	valueField = new SFVec2f(0.0f, 0.0f);
	addEventOut(valueFieldString, valueField);
}

CoordinateInterpolator2DNode::~CoordinateInterpolator2DNode() 
{
}

////////////////////////////////////////////////
//	keyValue
////////////////////////////////////////////////

MFVec2f *CoordinateInterpolator2DNode::getKeyValueField() const
{
	if (isInstanceNode() == false)
		return keyValueField;
	return (MFVec2f *)getExposedField(keyValueFieldString);
}
	
void CoordinateInterpolator2DNode::addKeyValue(float vector[]) 
{
	getKeyValueField()->addValue(vector);
}

int CoordinateInterpolator2DNode::getNKeyValues() const 
{
	return getKeyValueField()->getSize();
}
	
void CoordinateInterpolator2DNode::getKeyValue(int index, float vector[]) const 
{
	getKeyValueField()->get1Value(index, vector);
}

////////////////////////////////////////////////
//	value
////////////////////////////////////////////////

SFVec2f *CoordinateInterpolator2DNode::getValueField() const
{
	if (isInstanceNode() == false)
		return valueField;
	return (SFVec2f *)getEventOut(valueFieldString);
}
	
void CoordinateInterpolator2DNode::setValue(float vector[]) 
{
	getValueField()->setValue(vector);
}

void CoordinateInterpolator2DNode::getValue(float vector[]) const 
{
	getValueField()->getValue(vector);
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool CoordinateInterpolator2DNode::isChildNodeType(Node *node) const
{
	return false;
}

void CoordinateInterpolator2DNode::initialize() 
{
}

void CoordinateInterpolator2DNode::uninitialize() 
{
}

void CoordinateInterpolator2DNode::update() 
{
	int n;

	float fraction = getFraction();
	int index = -1;
	int nKey = getNKeys();
	for (n=0; n<(nKey-1); n++) {
		if (getKey(n) <= fraction && fraction <= getKey(n+1)) {
			index = n;
			break;
		}
	}
	if (index == -1)
		return;

	float scale = (fraction - getKey(index)) / (getKey(index+1) - getKey(index));

	float vector1[2];
	float vector2[2];
	float vectorOut[2];

	getKeyValue(index, vector1);
	getKeyValue(index+1, vector2);
	for (n=0; n<2; n++)
		vectorOut[n] = vector1[n] + (vector2[n] - vector1[n])*scale;

	setValue(vectorOut);
	sendEvent(getValueField());
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void CoordinateInterpolator2DNode::outputContext(std::ostream &printStream, const char *indentString) const 
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

CoordinateInterpolator2DNode *CoordinateInterpolator2DNode::next() const 
{
	return (CoordinateInterpolator2DNode *)Node::next(getType());
}

CoordinateInterpolator2DNode *CoordinateInterpolator2DNode::nextTraversal() const 
{
	return (CoordinateInterpolator2DNode *)Node::nextTraversalByType(getType());
}


