/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NormalInterpolatorNode.cpp
*
******************************************************************/

#include <x3d/VRML97Fields.h>
#include <x3d/NormalInterpolatorNode.h>

using namespace CyberX3D;

NormalInterpolatorNode::NormalInterpolatorNode() 
{
	setHeaderFlag(false);
	setType(NORMALINTERPOLATOR_NODE);

	// keyValue exposed field
	keyValueField = new MFVec3f();
	addExposedField(keyValueFieldString, keyValueField);

	// value_changed eventOut field
	valueField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addEventOut(valueFieldString, valueField);
}

NormalInterpolatorNode::~NormalInterpolatorNode() 
{
}

////////////////////////////////////////////////
//	keyValue
////////////////////////////////////////////////

MFVec3f *NormalInterpolatorNode::getKeyValueField() const
{
	if (isInstanceNode() == false)
		return keyValueField;
	return (MFVec3f *)getExposedField(keyValueFieldString);
}
	
void NormalInterpolatorNode::addKeyValue(float vector[]) 
{
	getKeyValueField()->addValue(vector);
}

int NormalInterpolatorNode::getNKeyValues() const
{
	return getKeyValueField()->getSize();
}
	
void NormalInterpolatorNode::getKeyValue(int index, float vector[]) const
{
	getKeyValueField()->get1Value(index, vector);
}

////////////////////////////////////////////////
//	value
////////////////////////////////////////////////

SFVec3f *NormalInterpolatorNode::getValueField() const
{
	if (isInstanceNode() == false)
		return valueField;
	return (SFVec3f *)getEventOut(valueFieldString);
}
	
void NormalInterpolatorNode::setValue(float vector[]) 
{
	getValueField()->setValue(vector);
}

void NormalInterpolatorNode::getValue(float vector[]) const
{
	getValueField()->getValue(vector);
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool NormalInterpolatorNode::isChildNodeType(Node *node) const
{
	return false;
}

void NormalInterpolatorNode::initialize() 
{
}

void NormalInterpolatorNode::uninitialize() 
{
}

void NormalInterpolatorNode::update() 
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

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void NormalInterpolatorNode::outputContext(std::ostream &printStream, const char *indentString) const
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

NormalInterpolatorNode *NormalInterpolatorNode::next() const
{
	return (NormalInterpolatorNode *)Node::next(getType());
}

NormalInterpolatorNode *NormalInterpolatorNode::nextTraversal() const
{
	return (NormalInterpolatorNode *)Node::nextTraversalByType(getType());
}

