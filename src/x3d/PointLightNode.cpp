/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PointLightNode.h
*
*	Revisions:
*
*	12/05/02
*		- Removed a ambientIntensity field.
*		- Removed getDiffuseColor(), getAmbientColor()
*
******************************************************************/

#include <x3d/PointLightNode.h>

using namespace CyberX3D;

PointLightNode::PointLightNode() 
{
	setHeaderFlag(false);
	setType(POINTLIGHT_NODE);

	// location exposed field
	locationField = new SFVec3f(0.0f, 0.0f, 0.0f);
	locationField->setName(locationFieldString);
	addExposedField(locationField);

	// radius exposed field
	radiusField = new SFFloat(100.0f);
	radiusField->setName(radiusFieldString);
	addExposedField(radiusField);

	// attenuation exposed field
	attenuationField = new SFVec3f(1.0f, 0.0f, 0.0f);
	attenuationField->setName(attenuationFieldString);
	addExposedField(attenuationField);
}

PointLightNode::~PointLightNode() 
{
}

////////////////////////////////////////////////
//	Location
////////////////////////////////////////////////

SFVec3f *PointLightNode::getLocationField() const
{
	if (isInstanceNode() == false)
		return locationField;
	return (SFVec3f *)getExposedField(locationFieldString);
}

void PointLightNode::setLocation(float value[]) 
{
	getLocationField()->setValue(value);
}

void PointLightNode::setLocation(float x, float y, float z) 
{
	getLocationField()->setValue(x, y, z);
}

void PointLightNode::getLocation(float value[]) const
{
	getLocationField()->getValue(value);
}

////////////////////////////////////////////////
//	Radius
////////////////////////////////////////////////

SFFloat *PointLightNode::getRadiusField() const
{
	if (isInstanceNode() == false)
		return radiusField;
	return (SFFloat *)getExposedField(radiusFieldString);
}
	
void PointLightNode::setRadius(float value) 
{
	getRadiusField()->setValue(value);
}

float PointLightNode::getRadius() const
{
	return getRadiusField()->getValue();
}

////////////////////////////////////////////////
//	Attenuation
////////////////////////////////////////////////

SFVec3f *PointLightNode::getAttenuationField() const
{
	if (isInstanceNode() == false)
		return attenuationField;
	return (SFVec3f *)getExposedField(attenuationFieldString);
}

void PointLightNode::setAttenuation(float value[]) 
{
	getAttenuationField()->setValue(value);
}

void PointLightNode::setAttenuation(float x, float y, float z) 
{
	getAttenuationField()->setValue(x, y, z);
}

void PointLightNode::getAttenuation(float value[]) const
{
	getAttenuationField()->getValue(value);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

PointLightNode *PointLightNode::next() const
{
	return (PointLightNode *)Node::next(getType());
}

PointLightNode *PointLightNode::nextTraversal() const
{
	return (PointLightNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool PointLightNode::isChildNodeType(Node *node) const
{
	return false;
}

void PointLightNode::initialize() 
{
}

void PointLightNode::uninitialize() 
{
}

void PointLightNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void PointLightNode::outputContext(std::ostream &printStream, const char *indentString) const 
{
	SFColor *color = getColorField();
	SFVec3f *attenuation = getAttenuationField();
	SFVec3f *location = getLocationField();
	SFBool *bon = getOnField();

	printStream << indentString << "\t" << "on " << bon  << std::endl;
	printStream << indentString << "\t" << "intensity " << getIntensity()  << std::endl;
	printStream << indentString << "\t" << "ambientIntensity " << getAmbientIntensity()  << std::endl;
	printStream << indentString << "\t" << "color " << color  << std::endl;
	printStream << indentString << "\t" << "location " << location  << std::endl;
	printStream << indentString << "\t" << "radius " << getRadius()  << std::endl;
	printStream << indentString << "\t" << "attenuation " << attenuation  << std::endl;
}
