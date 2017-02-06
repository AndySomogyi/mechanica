/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	DirectionalLightNode.cpp
*
*	Revisions:
*
*	12/05/02
*		- Removed a ambientIntensity field.
*		- Removed getDiffuseColor(), getAmbientColor()
*
******************************************************************/

#include <x3d/DirectionalLightNode.h>

using namespace CyberX3D;

DirectionalLightNode::DirectionalLightNode() 
{
	setType(DIRECTIONALLIGHT_NODE);

	// direction exposed field
	directionField = new SFVec3f(0.0f, 0.0f, -1.0f);
	directionField->setName(directionFieldString);
	addExposedField(directionField);
}

DirectionalLightNode::~DirectionalLightNode() 
{
}

////////////////////////////////////////////////
//	Direction
////////////////////////////////////////////////

SFVec3f *DirectionalLightNode::getDirectionField() const
{
	if (isInstanceNode() == false)
		return directionField;
	return (SFVec3f *)getExposedField(directionFieldString);
}

void DirectionalLightNode::setDirection(float value[]) 
{
	getDirectionField()->setValue(value);
}

void DirectionalLightNode::setDirection(float x, float y, float z)
{
	getDirectionField()->setValue(x, y, z);
}

void DirectionalLightNode::getDirection(float value[]) const 
{
	getDirectionField()->getValue(value);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

DirectionalLightNode *DirectionalLightNode::next() const 
{
	return (DirectionalLightNode *)Node::next(getType());
}

DirectionalLightNode *DirectionalLightNode::nextTraversal() const 
{
	return (DirectionalLightNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool DirectionalLightNode::isChildNodeType(Node *node) const
{
	return false;
}

void DirectionalLightNode::initialize() 
{
}

void DirectionalLightNode::uninitialize() 
{
}

void DirectionalLightNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void DirectionalLightNode::outputContext(std::ostream &printStream, const char *indentString) const 
{
	const SFBool *bon = getOnField();
	const SFVec3f *direction = getDirectionField();
	const SFColor *color = getColorField();

	printStream << indentString << "\t" << "on " << bon  << std::endl;
	printStream << indentString << "\t" << "intensity " << getIntensity()  << std::endl;
	printStream << indentString << "\t" << "ambientIntensity " << getAmbientIntensity()  << std::endl;
	printStream << indentString << "\t" << "color " << color  << std::endl;
	printStream << indentString << "\t" << "direction " << direction  << std::endl;
}
