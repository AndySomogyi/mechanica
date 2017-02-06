/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PositionInterpolatorNode.cpp
*
******************************************************************/

#include <x3d/VRML97Fields.h>
#include <x3d/PositionInterpolatorNode.h>

using namespace CyberX3D;

PositionInterpolatorNode::PositionInterpolatorNode() 
{
	setHeaderFlag(false);
	setType(POSITIONINTERPOLATOR_NODE);

	// keyValue exposed field
	keyValueField = new MFVec3f();
	addExposedField(keyValueFieldString, keyValueField);

	// value_changed eventOut field
	valueField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addEventOut(valueFieldString, valueField);
}

PositionInterpolatorNode::~PositionInterpolatorNode() 
{
}

////////////////////////////////////////////////
//	keyValue
////////////////////////////////////////////////
	
MFVec3f *PositionInterpolatorNode::getKeyValueField() const
{
	if (isInstanceNode() == false)
		return keyValueField;
	return (MFVec3f *)getExposedField(keyValueFieldString);
}

void PositionInterpolatorNode::addKeyValue(float vector[]) 
{
	getKeyValueField()->addValue(vector);
}

int PositionInterpolatorNode::getNKeyValues() const
{
	return getKeyValueField()->getSize();
}
	
void PositionInterpolatorNode::getKeyValue(int index, float vector[]) const
{
	getKeyValueField()->get1Value(index, vector);
}

////////////////////////////////////////////////
//	value
////////////////////////////////////////////////

SFVec3f *PositionInterpolatorNode::getValueField() const
{
	if (isInstanceNode() == false)
		return valueField;
	return (SFVec3f *)getEventOut(valueFieldString);
}
	
void PositionInterpolatorNode::setValue(float vector[]) 
{
	getValueField()->setValue(vector);
}

void PositionInterpolatorNode::getValue(float vector[]) const
{
	getValueField()->getValue(vector);
}

////////////////////////////////////////////////
//	Virtual functions
////////////////////////////////////////////////
	
bool PositionInterpolatorNode::isChildNodeType(Node *node) const
{
	return false;
}

void PositionInterpolatorNode::initialize() 
{
}

void PositionInterpolatorNode::uninitialize() 
{
}

void PositionInterpolatorNode::update() 
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

	float vector1[3];
	float vector2[3];
	float vectorOut[3];

	getKeyValue(index, vector1);
	getKeyValue(index+1, vector2);
	for (n=0; n<3; n++)
		vectorOut[n] = vector1[n] + (vector2[n] - vector1[n])*scale;

	setValue(vectorOut);
	sendEvent(getValueField());
}

void PositionInterpolatorNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	if (0 < getNKeys()) {
		MFFloat *key = getKeyField();
		printStream << indentString << "\tkey [" << std::endl;
		key->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t]" << std::endl;
	}
		
	if (0 < getNKeyValues()) {
		MFVec3f *keyValue = getKeyValueField();
		printStream << indentString << "\tkeyValue [" << std::endl;
		keyValue->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t]" << std::endl;
	}
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

PositionInterpolatorNode *PositionInterpolatorNode::next() const
{
	return (PositionInterpolatorNode *)Node::next(getType());
}

PositionInterpolatorNode *PositionInterpolatorNode::nextTraversal() const
{
	return (PositionInterpolatorNode *)Node::nextTraversalByType(getType());
}
