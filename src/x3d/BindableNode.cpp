/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BindableNode.cpp
*
******************************************************************/

#include <x3d/BindableNode.h>

using namespace CyberX3D;

BindableNode::BindableNode() 
{
	// set_bind
	setBindField = new SFBool(true);
	addEventIn(setBindFieldString, setBindField);

	// cybleInterval exposed field
	bindTimeField = new SFTime(1.0);
	addEventOut(bindTimeFieldString, bindTimeField);

	// isBind
	isBoundField = new SFBool(true);
	addEventOut(isBoundFieldString, isBoundField);
}

BindableNode::~BindableNode() 
{
}

////////////////////////////////////////////////
//	bind
////////////////////////////////////////////////

SFBool *BindableNode::getBindField() const
{
	if (isInstanceNode() == false)
		return setBindField;
	return (SFBool *)getEventIn(setBindFieldString);
}

void BindableNode::setBind(bool value) 
{
	getBindField()->setValue(value);
}

bool BindableNode::getBind() const
{
	return getBindField()->getValue();
}

bool BindableNode::isBind() const 
{
	return getBind();
}

////////////////////////////////////////////////
//	bindTime
////////////////////////////////////////////////

SFTime *BindableNode::getBindTimeField() const
{
	if (isInstanceNode() == false)
		return bindTimeField;
	return (SFTime *)getEventOut(bindTimeFieldString);
}

void BindableNode::setBindTime(double value) 
{
	getBindTimeField()->setValue(value);
}

double BindableNode::getBindTime() const
{
	return getBindTimeField()->getValue();
}

////////////////////////////////////////////////
//	isBound
////////////////////////////////////////////////

SFBool *BindableNode::getIsBoundField() const
{
	if (isInstanceNode() == false)
		return isBoundField;
	return (SFBool *)getEventOut(isBoundFieldString);
}

void BindableNode::setIsBound(bool value) 
{
	getIsBoundField()->setValue(value);
}

bool BindableNode::getIsBound() const
{
	return getIsBoundField()->getValue();
}

bool BindableNode::isBound() const
{
	return getIsBound();
}

