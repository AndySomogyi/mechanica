/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TimeTriggerNode.cpp
*
******************************************************************/

#include <x3d/TimeTriggerNode.h>

using namespace CyberX3D;

static const char integerKeyFieldString[] = "integerKey";
static const char set_booleanFieldString[] = "set_boolean";
static const char triggerTimeFieldString[] = "triggerTime";

TimeTriggerNode::TimeTriggerNode() 
{
	setHeaderFlag(false);
	setType(TIMETRIGGER_NODE);

	// set_boolean eventIn field
	set_booleanField = new SFBool(false);
	addEventIn(set_booleanFieldString, set_booleanField);

	// triggerTime eventIOut field
	triggerTimeField = new SFTime();
	addEventOut(triggerTimeFieldString, triggerTimeField);
}

TimeTriggerNode::~TimeTriggerNode() 
{
}

////////////////////////////////////////////////
//	setBoolean
////////////////////////////////////////////////

SFBool *TimeTriggerNode::getBooleanField() const
{
	if (isInstanceNode() == false)
		return set_booleanField;
	return (SFBool*)getEventIn(set_booleanFieldString);
}
	
void TimeTriggerNode::setBoolean(bool value) 
{
	getBooleanField()->setValue(value);
}

bool TimeTriggerNode::getBoolean() const
{
	return getBooleanField()->getValue();
}

bool TimeTriggerNode::isBoolean() const
{
	return getBooleanField()->getValue();
}

////////////////////////////////////////////////
//	TriggerTime
////////////////////////////////////////////////

SFTime *TimeTriggerNode::getTriggerTimeField() const
{
	if (isInstanceNode() == false)
		return triggerTimeField;
	return (SFTime*)getEventOut(triggerTimeFieldString);
}
	
void TimeTriggerNode::setTriggerTime(double value) 
{
	getTriggerTimeField()->setValue(value);
}

double TimeTriggerNode::getTriggerTime() const
{
	return getTriggerTimeField()->getValue();
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool TimeTriggerNode::isChildNodeType(Node *node) const
{
	return false;
}

void TimeTriggerNode::initialize() 
{
}

void TimeTriggerNode::uninitialize() 
{
}

void TimeTriggerNode::update() 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void TimeTriggerNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

TimeTriggerNode *TimeTriggerNode::next() const
{
	return (TimeTriggerNode *)Node::next(getType());
}

TimeTriggerNode *TimeTriggerNode::nextTraversal() const
{
	return (TimeTriggerNode *)Node::nextTraversalByType(getType());
}




