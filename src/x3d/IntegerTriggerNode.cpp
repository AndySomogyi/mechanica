/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	IntegerTriggerNode.cpp
*
******************************************************************/

#include <x3d/IntegerTriggerNode.h>

using namespace CyberX3D;

static const char integerKeyFieldString[] = "integerKey";
static const char set_booleanFieldString[] = "set_boolean";
static const char triggerValueFieldString[] = "triggerValue";

IntegerTriggerNode::IntegerTriggerNode() 
{
	setHeaderFlag(false);
	setType(INTEGERTRIGGER_NODE);

	// set_boolean eventIn field
	set_booleanField = new SFBool(false);
	addEventIn(set_booleanFieldString, set_booleanField);

	// integerKey exposed field
	integerKeyField = new SFInt32(-1);
	addExposedField(integerKeyFieldString, integerKeyField);

	// triggerValue eventIn field
	triggerValueField = new SFInt32();
	addEventOut(triggerValueFieldString, triggerValueField);
}

IntegerTriggerNode::~IntegerTriggerNode() 
{
}

////////////////////////////////////////////////
//	setBoolean
////////////////////////////////////////////////

SFBool *IntegerTriggerNode::getBooleanField() const
{
	if (isInstanceNode() == false)
		return set_booleanField;
	return (SFBool*)getEventIn(set_booleanFieldString);
}
	
void IntegerTriggerNode::setBoolean(bool value) 
{
	getBooleanField()->setValue(value);
}

bool IntegerTriggerNode::getBoolean() const
{
	return getBooleanField()->getValue();
}

bool IntegerTriggerNode::isBoolean() const
{
	return getBooleanField()->getValue();
}

////////////////////////////////////////////////
//	IntegerKey
////////////////////////////////////////////////

SFInt32 *IntegerTriggerNode::getIntegerKeyField() const
{
	if (isInstanceNode() == false)
		return integerKeyField;
	return (SFInt32*)getExposedField(integerKeyFieldString);
}
	
void IntegerTriggerNode::setIntegerKey(int value) 
{
	getIntegerKeyField()->setValue(value);
}

int IntegerTriggerNode::getIntegerKey() const
{
	return getIntegerKeyField()->getValue();
}

////////////////////////////////////////////////
//	TriggerValue
////////////////////////////////////////////////

SFInt32 *IntegerTriggerNode::getTriggerValueField() const
{
	if (isInstanceNode() == false)
		return triggerValueField;
	return (SFInt32*)getEventOut(triggerValueFieldString);
}
	
void IntegerTriggerNode::setTriggerValue(int value) 
{
	getTriggerValueField()->setValue(value);
}

int IntegerTriggerNode::getTriggerValue() const
{
	return getTriggerValueField()->getValue();
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool IntegerTriggerNode::isChildNodeType(Node *node) const
{
	return false;
}

void IntegerTriggerNode::initialize() 
{
}

void IntegerTriggerNode::uninitialize() 
{
}

void IntegerTriggerNode::update() 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void IntegerTriggerNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

IntegerTriggerNode *IntegerTriggerNode::next() const
{
	return (IntegerTriggerNode *)Node::next(getType());
}

IntegerTriggerNode *IntegerTriggerNode::nextTraversal() const
{
	return (IntegerTriggerNode *)Node::nextTraversalByType(getType());
}




