/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MultiTextureCoordinateNode.cpp
*
*	Revisions;
*
*	12/06/02
*		- The first revision.
*
******************************************************************/

#include <x3d/MultiTextureCoordinateNode.h>

using namespace CyberX3D;

static const char texCoordFieldString[] = "texCoord";

MultiTextureCoordinateNode::MultiTextureCoordinateNode() 
{
	setHeaderFlag(false);
	setType(MULTITEXTURECOORD_NODE);

	// texCoord exposed field
	texCoordField = new MFNode();
	addExposedField(texCoordFieldString, texCoordField);
}

MultiTextureCoordinateNode::~MultiTextureCoordinateNode() 
{
}

////////////////////////////////////////////////
//	texCoord
////////////////////////////////////////////////

MFNode *MultiTextureCoordinateNode::getTexCoordField() const
{
	if (isInstanceNode() == false)
		return texCoordField;
	return (MFNode *)getExposedField(texCoordFieldString);
}
	
void MultiTextureCoordinateNode::addTexCoord(Node *value) 
{
	getTexCoordField()->addValue(value);
}

int MultiTextureCoordinateNode::getNTexCoords() const
{
	return getTexCoordField()->getSize();
}
	
Node *MultiTextureCoordinateNode::getTexCoord(int index) const
{
	return getTexCoordField()->get1Value(index);
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool MultiTextureCoordinateNode::isChildNodeType(Node *node) const
{
	return false;
}

void MultiTextureCoordinateNode::initialize() 
{
}

void MultiTextureCoordinateNode::uninitialize() 
{
}

void MultiTextureCoordinateNode::update() 
{
/*
	float fraction = getFraction();
	int index = -1;
	int nKey = getNKeys();
	for (int n=0; n<(nKey-1); n++) {
		if (getKey(n) <= fraction && fraction <= getKey(n+1)) {
			index = n;
			break;
		}
	}
	if (index == -1)
		return;

	float scale = (fraction - getKey(index)) / (getKey(index+1) - getKey(index));

	float value1 = getTexCoord(index);
	float value2 = getTexCoord(index+1);
	float valueOut = value1 + (value2 - value1)*scale;

	setValue(valueOut);
	sendEvent(getValueField());
*/
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void MultiTextureCoordinateNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

MultiTextureCoordinateNode *MultiTextureCoordinateNode::next() const
{
	return (MultiTextureCoordinateNode *)Node::next(getType());
}

MultiTextureCoordinateNode *MultiTextureCoordinateNode::nextTraversal() const
{
	return (MultiTextureCoordinateNode *)Node::nextTraversalByType(getType());
}
