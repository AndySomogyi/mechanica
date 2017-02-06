/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	IntegerSequencerNode.cpp
*
******************************************************************/

#include <x3d/VRML97Fields.h>
#include <x3d/IntegerSequencerNode.h>

using namespace CyberX3D;

IntegerSequencerNode::IntegerSequencerNode() 
{
	setHeaderFlag(false);
	setType(INTEGERSEQUENCER_NODE);

	// keyValue exposed field
	keyValueField = new MFInt32();
	addExposedField(keyValueFieldString, keyValueField);

	// value_changed eventOut field
	valueField = new SFInt32();
	addEventOut(valueFieldString, valueField);
}

IntegerSequencerNode::~IntegerSequencerNode() 
{
}

////////////////////////////////////////////////
//	keyValue
////////////////////////////////////////////////

MFInt32 *IntegerSequencerNode::getKeyValueField() const
{
	if (isInstanceNode() == false)
		return keyValueField;
	return (MFInt32 *)getExposedField(keyValueFieldString);
}
	
void IntegerSequencerNode::addKeyValue(int value) 
{
	getKeyValueField()->addValue(value);
}

int IntegerSequencerNode::getNKeyValues() const
{
	return getKeyValueField()->getSize();
}
	
int IntegerSequencerNode::getKeyValue(int index) const
{
	return getKeyValueField()->get1Value(index);
}

////////////////////////////////////////////////
//	value
////////////////////////////////////////////////

SFInt32 *IntegerSequencerNode::getValueField() const
{
	if (isInstanceNode() == false)
		return valueField;
	return (SFInt32 *)getEventOut(valueFieldString);
}
	
void IntegerSequencerNode::setValue(int vector) 
{
	getValueField()->setValue(vector);
}

int IntegerSequencerNode::getValue() const
{
	return getValueField()->getValue();
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool IntegerSequencerNode::isChildNodeType(Node *node) const
{
	return false;
}

void IntegerSequencerNode::initialize() 
{
}

void IntegerSequencerNode::uninitialize() 
{
}

void IntegerSequencerNode::update() 
{
	float fraction = getFraction();
	int index = -1;
	int nKey = getNKeys();
	for (int n=0; n<(nKey-1); n++) {
		if (getKey(n) <= fraction && fraction <= getKey(n+1)) {
			index = n;
			break;
		}
	}
	if (index == -1)
		return;

	float scale = (fraction - getKey(index)) / (getKey(index+1) - getKey(index));

	float value1 = (float)getKeyValue(index);
	float value2 = (float)getKeyValue(index+1);
	float valueOut = value1 + (value2 - value1)*scale;

	setValue((int)valueOut);
	sendEvent(getValueField());
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void IntegerSequencerNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

IntegerSequencerNode *IntegerSequencerNode::next() const
{
	return (IntegerSequencerNode *)Node::next(getType());
}

IntegerSequencerNode *IntegerSequencerNode::nextTraversal() const
{
	return (IntegerSequencerNode *)Node::nextTraversalByType(getType());
}
