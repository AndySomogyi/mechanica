/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanTriggerNode.cpp
*
*	Revisions;
*
*	01/08/03
*		- first revision
*
******************************************************************/

#include <x3d/BooleanTriggerNode.h>

using namespace CyberX3D;

static const char set_triggerTimeFieldString[] = "set_triggerTime";
static const char triggerTrueFieldString[] = "triggerTrue";

BooleanTriggerNode::BooleanTriggerNode() 
{
	setHeaderFlag(false);
	setType(BOOLEANTRIGGER_NODE);

	// set_triggerTimeEvent eventIn field
	set_triggerTimeField = new SFTime();
	addEventIn(set_triggerTimeFieldString, set_triggerTimeField);

	// triggerTrue eventIn field
	triggerTrueField = new SFBool();
	addEventOut(triggerTrueFieldString, triggerTrueField);
}

BooleanTriggerNode::~BooleanTriggerNode() 
{
}

////////////////////////////////////////////////
//	triggerTime
////////////////////////////////////////////////

SFTime *BooleanTriggerNode::getTriggerTimeField() const
{
	if (isInstanceNode() == false)
		return set_triggerTimeField;
	return (SFTime*)getEventIn(set_triggerTimeFieldString);
}
	
void BooleanTriggerNode::setTriggerTime(double value) 
{
	getTriggerTimeField()->setValue(value);
}

double BooleanTriggerNode::getTriggerTime()  const
{
	return getTriggerTimeField()->getValue();
}

////////////////////////////////////////////////
//	triggerTrue
////////////////////////////////////////////////

SFBool *BooleanTriggerNode::getTriggerTrueField() const
{
	if (isInstanceNode() == false)
		return triggerTrueField;
	return (SFBool*)getEventOut(triggerTrueFieldString);
}
	
void BooleanTriggerNode::setTriggerTrue(bool value) 
{
	getTriggerTrueField()->setValue(value);
}

bool BooleanTriggerNode::getTriggerTrue()  const
{
	return getTriggerTrueField()->getValue();
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool BooleanTriggerNode::isChildNodeType(Node *node) const
{
	return false;
}

void BooleanTriggerNode::initialize() 
{
}

void BooleanTriggerNode::uninitialize() 
{
}

void BooleanTriggerNode::update() 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void BooleanTriggerNode::outputContext(std::ostream &printStream, const char *indentString)  const
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

BooleanTriggerNode *BooleanTriggerNode::next()  const
{
	return (BooleanTriggerNode *)Node::next(getType());
}

BooleanTriggerNode *BooleanTriggerNode::nextTraversal()  const
{
	return (BooleanTriggerNode *)Node::nextTraversalByType(getType());
}


