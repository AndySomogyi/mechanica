/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SequencerNode.cpp
*
******************************************************************/

#include <x3d/VRML97Fields.h>
#include <x3d/SequencerNode.h>

using namespace CyberX3D;

SequencerNode::SequencerNode() 
{
	// key exposed field
	keyField = new MFFloat();
	addExposedField(keyFieldString, keyField);

	// set_fraction eventIn field
	fractionField = new SFFloat(0.0f);
	addEventIn(fractionFieldString, fractionField);
}

SequencerNode::~SequencerNode() 
{
}

////////////////////////////////////////////////
//	key
////////////////////////////////////////////////

MFFloat *SequencerNode::getKeyField() const
{
	if (isInstanceNode() == false)
		return keyField;
	return (MFFloat *)getExposedField(keyFieldString);
}
	
void SequencerNode::addKey(float value) 
{
	getKeyField()->addValue(value);
}

int SequencerNode::getNKeys() const
{
	return getKeyField()->getSize();
}

float SequencerNode::getKey(int index) const
{
	return getKeyField()->get1Value(index);
}


////////////////////////////////////////////////
//	fraction
////////////////////////////////////////////////

SFFloat *SequencerNode::getFractionField() const
{
	if (isInstanceNode() == false)
		return fractionField;
	return (SFFloat *)getEventIn(fractionFieldString);
}
	
void SequencerNode::setFraction(float value) 
{
	getFractionField()->setValue(value);
}

float SequencerNode::getFraction() const
{
	return getFractionField()->getValue();
}
