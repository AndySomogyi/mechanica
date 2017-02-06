/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanFilterNode.cpp
*
*	Revisions;
*
*	01/08/03
*		- first revision
*
******************************************************************/

#include <x3d/BooleanFilterNode.h>

using namespace CyberX3D;

static const char set_booleanFieldString[] = "set_boolean";
static const char inputFalseFieldString[] = "inputFalse";
static const char inputNegateFieldString[] = "inputNegate";
static const char inputTrueFieldString[] = "inputTrue";

BooleanFilterNode::BooleanFilterNode() 
{
	setHeaderFlag(false);
	setType(BOOLEANFILTER_NODE);

	// set_booleanEvent eventIn field
	set_booleanField = new SFBool();
	addEventIn(set_booleanFieldString, set_booleanField);

	// inputFalse eventIn field
	inputFalseField = new SFBool();
	addEventOut(inputFalseFieldString, inputFalseField);

	// inputNegate eventIn field
	inputNegateField = new SFBool();
	addEventOut(inputNegateFieldString, inputNegateField);

	// inputTrue eventIn field
	inputTrueField = new SFBool();
	addEventOut(inputTrueFieldString, inputTrueField);
}

BooleanFilterNode::~BooleanFilterNode() 
{
}

////////////////////////////////////////////////
//	setBoolean
////////////////////////////////////////////////

SFBool *BooleanFilterNode::getBooleanField() const
{
	if (isInstanceNode() == false)
		return set_booleanField;
	return (SFBool*)getEventIn(set_booleanFieldString);
}
	
void BooleanFilterNode::setBoolean(bool value) 
{
	getBooleanField()->setValue(value);
}

bool BooleanFilterNode::getBoolean()  const
{
	return getBooleanField()->getValue();
}

bool BooleanFilterNode::isBoolean()  const
{
	return getBooleanField()->getValue();
}

////////////////////////////////////////////////
//	inputFalse
////////////////////////////////////////////////

SFBool *BooleanFilterNode::getInputFalseField() const
{
	if (isInstanceNode() == false)
		return inputFalseField;
	return (SFBool*)getEventOut(inputFalseFieldString);
}
	
void BooleanFilterNode::setInputFalse(bool value) 
{
	getInputFalseField()->setValue(value);
}

bool BooleanFilterNode::getInputFalse()  const
{
	return getInputFalseField()->getValue();
}

bool BooleanFilterNode::isInputFalse()  const
{
	return getInputFalseField()->getValue();
}

////////////////////////////////////////////////
//	inputNegate
////////////////////////////////////////////////

SFBool *BooleanFilterNode::getInputNegateField() const
{
	if (isInstanceNode() == false)
		return inputNegateField;
	return (SFBool*)getEventOut(inputNegateFieldString);
}
	
void BooleanFilterNode::setInputNegate(bool value) 
{
	getInputNegateField()->setValue(value);
}

bool BooleanFilterNode::getInputNegate()  const
{
	return getInputNegateField()->getValue();
}

bool BooleanFilterNode::isInputNegate()  const
{
	return getInputNegateField()->getValue();
}

////////////////////////////////////////////////
//	inputTrue
////////////////////////////////////////////////

SFBool *BooleanFilterNode::getInputTrueField() const
{
	if (isInstanceNode() == false)
		return inputTrueField;
	return (SFBool*)getEventOut(inputTrueFieldString);
}
	
void BooleanFilterNode::setInputTrue(bool value) 
{
	getInputTrueField()->setValue(value);
}

bool BooleanFilterNode::getInputTrue()  const
{
	return getInputTrueField()->getValue();
}

bool BooleanFilterNode::isInputTrue()  const
{
	return getInputTrueField()->getValue();
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool BooleanFilterNode::isChildNodeType(Node *node) const
{
	return false;
}

void BooleanFilterNode::initialize() 
{
}

void BooleanFilterNode::uninitialize() 
{
}

void BooleanFilterNode::update() 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void BooleanFilterNode::outputContext(std::ostream &printStream, const char *indentString)  const
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

BooleanFilterNode *BooleanFilterNode::next()  const
{
	return (BooleanFilterNode *)Node::next(getType());
}

BooleanFilterNode *BooleanFilterNode::nextTraversal()  const
{
	return (BooleanFilterNode *)Node::nextTraversalByType(getType());
}

