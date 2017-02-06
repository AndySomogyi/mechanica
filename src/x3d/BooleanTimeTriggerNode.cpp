/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanTimeTriggerNode.cpp
*
******************************************************************/

#include <x3d/BooleanTimeTriggerNode.h>

using namespace CyberX3D;

static const char set_booleanTrueFieldString[] = "set_booleanTrue";
static const char set_booleanFalseFieldString[] = "set_booleanFalse";
static const char trueTriggerFieldString[] = "trueTrigger";
static const char falseTriggerFieldString[] = "falseTrigger";

BooleanTimeTriggerNode::BooleanTimeTriggerNode() 
{
	setHeaderFlag(false);
	setType(BOOLEANTIMETRIGGER_NODE);

	// triggerTimeField eventIn field
	set_booleanTrueField = new SFBool(false);
	addEventIn(set_booleanTrueFieldString, set_booleanTrueField);

	// triggerTimeField eventIn field
	set_booleanFalseField = new SFBool(true);
	addEventIn(set_booleanFalseFieldString, set_booleanFalseField);

	// triggerTimeField eventOut field
	trueTriggerField = new SFBool();
	addEventOut(trueTriggerFieldString, trueTriggerField);

	// triggerTimeField eventOut field
	falseTriggerField = new SFBool();
	addEventOut(falseTriggerFieldString, falseTriggerField);
}

BooleanTimeTriggerNode::~BooleanTimeTriggerNode() 
{
}

////////////////////////////////////////////////
//	SetBooleanTrue
////////////////////////////////////////////////

SFBool *BooleanTimeTriggerNode::getSetBooleanTrueField() const
{
	if (isInstanceNode() == false)
		return set_booleanTrueField;
	return (SFBool*)getEventIn(set_booleanTrueFieldString);
}
	
void BooleanTimeTriggerNode::setSetBooleanTrue(bool value) 
{
	getSetBooleanTrueField()->setValue(value);
}

bool BooleanTimeTriggerNode::getSetBooleanTrue() const
{
	return getSetBooleanTrueField()->getValue();
}

////////////////////////////////////////////////
//	SetBooleanFalse
////////////////////////////////////////////////

SFBool *BooleanTimeTriggerNode::getSetBooleanFalseField() const
{
	if (isInstanceNode() == false)
		return set_booleanFalseField;
	return (SFBool*)getEventIn(set_booleanFalseFieldString);
}
	
void BooleanTimeTriggerNode::setSetBooleanFalse(bool value) 
{
	getSetBooleanFalseField()->setValue(value);
}

bool BooleanTimeTriggerNode::getSetBooleanFalse() const
{
	return getSetBooleanFalseField()->getValue();
}

////////////////////////////////////////////////
//	TrueTrigger
////////////////////////////////////////////////

SFBool *BooleanTimeTriggerNode::getTrueTriggerField() const
{
	if (isInstanceNode() == false)
		return trueTriggerField;
	return (SFBool*)getEventOut(trueTriggerFieldString);
}
	
void BooleanTimeTriggerNode::setTrueTrigger(bool value) 
{
	getTrueTriggerField()->setValue(value);
}

bool BooleanTimeTriggerNode::getTrueTrigger() const
{
	return getTrueTriggerField()->getValue();
}

////////////////////////////////////////////////
//	FalseTrigger
////////////////////////////////////////////////

SFBool *BooleanTimeTriggerNode::getFalseTriggerField() const
{
	if (isInstanceNode() == false)
		return falseTriggerField;
	return (SFBool*)getEventOut(falseTriggerFieldString);
}
	
void BooleanTimeTriggerNode::setFalseTrigger(bool value) 
{
	getFalseTriggerField()->setValue(value);
}

bool BooleanTimeTriggerNode::getFalseTrigger() const
{
	return getFalseTriggerField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

BooleanTimeTriggerNode *BooleanTimeTriggerNode::next() const
{
	return (BooleanTimeTriggerNode *)Node::next(getType());
}

BooleanTimeTriggerNode *BooleanTimeTriggerNode::nextTraversal() const 
{
	return (BooleanTimeTriggerNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool BooleanTimeTriggerNode::isChildNodeType(Node *node) const
{
	return false;
}

void BooleanTimeTriggerNode::initialize() 
{
}

void BooleanTimeTriggerNode::uninitialize() 
{
}

void BooleanTimeTriggerNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void BooleanTimeTriggerNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}
