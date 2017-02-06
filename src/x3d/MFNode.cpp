/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFNode.cpp
*
******************************************************************/

#include <x3d/MFNode.h>

using namespace CyberX3D;

MFNode::MFNode() 
{
	setType(fieldTypeMFNode);
}

void MFNode::addValue(Node *value) 
{
	SFNode *sfvalue = new SFNode(value);
	add(sfvalue);
}

void MFNode::addValue(SFNode *sfvalue) 
{
	add(sfvalue);
}

void MFNode::addValue(const char *value) 
{
}

void MFNode::insertValue(int index, Node *value) 
{
	SFNode *sfvalue = new SFNode(value);
	insert(sfvalue, index);
}

Node *MFNode::get1Value(int index) const
{
	SFNode *sfvalue = (SFNode *)getObject(index);
	if (sfvalue)
		return sfvalue->getValue();
	return NULL;
}

void MFNode::set1Value(int index, Node *value) 
{
	SFNode *sfvalue = (SFNode *)getObject(index);
	if (sfvalue)
		sfvalue->setValue(value);
}

void MFNode::setValue(MFNode *values)
{
	clear();

	int size = values->getSize();
	for (int n=0; n<size; n++)
		addValue(values->get1Value(n));
}

void MFNode::setValue(MField *mfield)
{
	if (mfield->getType() == fieldTypeMFNode)
		setValue((MFNode *)mfield);
}

void MFNode::setValue(int size, Node *values[])
{
	clear();

	for (int n=0; n<size; n++)
		addValue(values[n]);
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void MFNode::outputContext(std::ostream& printStream, const char *indentString) const
{
}
