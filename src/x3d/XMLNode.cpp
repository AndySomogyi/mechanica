/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C)Satoshi Konno 1996-2007
*
*	File:	XMLNode.cpp
*
******************************************************************/

#include <x3d/XMLNode.h>
#include <x3d/TextureNode.h>

using namespace CyberX3D;

XMLNode::XMLNode()
{
	setHeaderFlag(false);
	setType(XML_NODE);
}

XMLNode::~XMLNode()
{
}

////////////////////////////////////////////////
//	Element Field
////////////////////////////////////////////////

XMLElement *XMLNode::getElement(const char *eleName) const
{
	return (XMLElement *)getField(eleName);
}
	
int XMLNode::getNElements() const
{
	return getNFields();
}

void XMLNode::addElement(XMLElement *ele)
{
	addField(ele);
}

void XMLNode::addElement(const char *name, XMLElement *ele)
{
	addField(name, ele);
}

void XMLNode::addElement(const char *name, const char *value)
{
	XMLElement *ele = new XMLElement(value);
	addField(name, ele);
}

XMLElement *XMLNode::getElement(int index) const
{
	return (XMLElement *)getField(index);
}

/*
bool XMLNode::removeElement(XMLElement *ele)
{
	return removeField(ele);
}

bool XMLNode::removeElement(const char *eleName)
{
	return removeField(eleName);
}

int XMLNode::getElementNumber(XMLElement ele)
{
	return getFieldNumber(ele);
}
*/

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

XMLNode *XMLNode::next() const
{
	return (XMLNode *)Node::next(getType());
}

XMLNode *XMLNode::nextTraversal() const
{
	return (XMLNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	virtual functions
////////////////////////////////////////////////
	
bool XMLNode::isChildNodeType(Node *node) const
{
	return true;
}

void XMLNode::initialize()
{
}

void XMLNode::uninitialize()
{
}

void XMLNode::update()
{
}

void XMLNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}
