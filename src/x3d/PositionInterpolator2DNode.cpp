/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PositionInterpolator2DNode.cpp
*
******************************************************************/

#include <x3d/X3DFields.h>
#include <x3d/PositionInterpolator2DNode.h>

using namespace CyberX3D;

PositionInterpolator2DNode::PositionInterpolator2DNode() 
{
	setHeaderFlag(false);
	setType(POSITIONINTERPOLATOR2D_NODE);

	// keyValue exposed field
	keyValueField = new MFVec2f();
	addExposedField(keyValueFieldString, keyValueField);

	// value_changed eventOut field
	valueField = new SFVec2f(0.0f, 0.0f);
	addEventOut(valueFieldString, valueField);
}

PositionInterpolator2DNode::~PositionInterpolator2DNode() 
{
}

////////////////////////////////////////////////
//	keyValue
////////////////////////////////////////////////
	
MFVec2f *PositionInterpolator2DNode::getKeyValueField() const
{
	if (isInstanceNode() == false)
		return keyValueField;
	return (MFVec2f *)getExposedField(keyValueFieldString);
}

void PositionInterpolator2DNode::addKeyValue(float vector[]) 
{
	getKeyValueField()->addValue(vector);
}

int PositionInterpolator2DNode::getNKeyValues() const
{
	return getKeyValueField()->getSize();
}
	
void PositionInterpolator2DNode::getKeyValue(int index, float vector[]) const
{
	getKeyValueField()->get1Value(index, vector);
}

////////////////////////////////////////////////
//	value
////////////////////////////////////////////////

SFVec2f *PositionInterpolator2DNode::getValueField() const
{
	if (isInstanceNode() == false)
		return valueField;
	return (SFVec2f *)getEventOut(valueFieldString);
}
	
void PositionInterpolator2DNode::setValue(float vector[]) 
{
	getValueField()->setValue(vector);
}

void PositionInterpolator2DNode::getValue(float vector[]) const
{
	getValueField()->getValue(vector);
}

////////////////////////////////////////////////
//	Virtual functions
////////////////////////////////////////////////
	
bool PositionInterpolator2DNode::isChildNodeType(Node *node) const
{
	return false;
}

void PositionInterpolator2DNode::initialize() 
{
}

void PositionInterpolator2DNode::uninitialize() 
{
}

void PositionInterpolator2DNode::update() 
{
	int	n;

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

void PositionInterpolator2DNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

PositionInterpolator2DNode *PositionInterpolator2DNode::next() const
{
	return (PositionInterpolator2DNode *)Node::next(getType());
}

PositionInterpolator2DNode *PositionInterpolator2DNode::nextTraversal() const
{
	return (PositionInterpolator2DNode *)Node::nextTraversalByType(getType());
}
