/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanSequencerNode.cpp
*
******************************************************************/

#include <x3d/BooleanSequencerNode.h>

using namespace CyberX3D;

BooleanSequencerNode::BooleanSequencerNode() 
{
	setHeaderFlag(false);
	setType(BOOLEANSEQUENCER_NODE);

	// keyValue exposed field
	keyValueField = new MFBool();
	addExposedField(keyValueFieldString, keyValueField);

	// value_changed eventOut field
	valueField = new SFBool();
	addEventOut(valueFieldString, valueField);
}

BooleanSequencerNode::~BooleanSequencerNode() 
{
}

////////////////////////////////////////////////
//	keyValue
////////////////////////////////////////////////

MFBool *BooleanSequencerNode::getKeyValueField() const
{
	if (isInstanceNode() == false)
		return keyValueField;
	return (MFBool *)getExposedField(keyValueFieldString);
}
	
void BooleanSequencerNode::addKeyValue(bool value) 
{
	getKeyValueField()->addValue(value);
}

int BooleanSequencerNode::getNKeyValues() const
{
	return getKeyValueField()->getSize();
}
	
bool BooleanSequencerNode::getKeyValue(int index) const
{
	return getKeyValueField()->get1Value(index);
}

////////////////////////////////////////////////
//	value
////////////////////////////////////////////////

SFBool *BooleanSequencerNode::getValueField() const
{
	if (isInstanceNode() == false)
		return valueField;
	return (SFBool *)getEventOut(valueFieldString);
}
	
void BooleanSequencerNode::setValue(bool vector) 
{
	getValueField()->setValue(vector);
}

bool BooleanSequencerNode::getValue() const
{
	return getValueField()->getValue();
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool BooleanSequencerNode::isChildNodeType(Node *node) const
{
	return false;
}

void BooleanSequencerNode::initialize() 
{
}

void BooleanSequencerNode::uninitialize() 
{
}

void BooleanSequencerNode::update() 
{
/*
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

	float value1 = getKeyValue(index);
	float value2 = getKeyValue(index+1);
	float valueOut = value1 + (value2 - value1)*scale;

	setValue(valueOut);
	sendEvent(getValueField());
*/
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void BooleanSequencerNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

BooleanSequencerNode *BooleanSequencerNode::next() const
{
	return (BooleanSequencerNode *)Node::next(getType());
}

BooleanSequencerNode *BooleanSequencerNode::nextTraversal() const
{
	return (BooleanSequencerNode *)Node::nextTraversalByType(getType());
}

