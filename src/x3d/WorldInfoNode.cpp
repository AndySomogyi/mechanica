/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: WorldInfoNode.cpp
*
*	11/20/02
*		- Changed the super class from Node to InfoNode.
*
******************************************************************/

#include <x3d/WorldInfoNode.h>

using namespace CyberX3D;

WorldInfoNode::WorldInfoNode() 
{
	setHeaderFlag(false);
	setType(WORLDINFO_NODE);

	// title exposed field
	titleField = new SFString("");
	addField(titleFieldString, titleField);

	// info exposed field
	infoField = new MFString();
	addField(infoFieldString, infoField);
}

WorldInfoNode::~WorldInfoNode()
{
}

////////////////////////////////////////////////
//	Title
////////////////////////////////////////////////

SFString *WorldInfoNode::getTitleField() const
{
	if (isInstanceNode() == false)
		return titleField;
	return (SFString *)getField(titleFieldString);
}
	
void WorldInfoNode::setTitle(const char *value)  
{
	getTitleField()->setValue(value);
}

const char *WorldInfoNode::getTitle() const
{
	return getTitleField()->getValue();
}

////////////////////////////////////////////////
// Info
////////////////////////////////////////////////

MFString *WorldInfoNode::getInfoField() const
{
	if (isInstanceNode() == false)
		return infoField;
	return (MFString *)getField(infoFieldString);
}

void WorldInfoNode::addInfo(const char *value) 
{
	getInfoField()->addValue(value);
}

int WorldInfoNode::getNInfos() const
{
	return getInfoField()->getSize();
}

const char *WorldInfoNode::getInfo(int index) const
{
	return getInfoField()->get1Value(index);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

WorldInfoNode *WorldInfoNode::next() const 
{
	return (WorldInfoNode *)Node::next(getType());
}

WorldInfoNode *WorldInfoNode::nextTraversal() const
{
	return (WorldInfoNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool WorldInfoNode::isChildNodeType(Node *node) const
{
	return false;
}

void WorldInfoNode::initialize() 
{
}

void WorldInfoNode::uninitialize() 
{
}

void WorldInfoNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void WorldInfoNode::outputContext(std::ostream& printStream, const char *indentString) const
{
	SFString *title = getTitleField();
	printStream << indentString << "\t" << "title " << title << std::endl;

	if (0 < getNInfos()) {
		MFString *info = getInfoField();
		printStream <<  indentString << "\t" << "info ["  << std::endl;
		info->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}
