/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SpotLightNode.cpp
*
*	Revisions:
*
*	12/05/02
*		- Removed a ambientIntensity field.
*		- Removed getDiffuseColor(), getAmbientColor()
*
******************************************************************/

#include <x3d/SpotLightNode.h>

using namespace CyberX3D;

SpotLightNode::SpotLightNode() 
{
	setType(SPOTLIGHT_NODE);

	// location exposed field
	locationField = new SFVec3f(0.0f, 0.0f, 0.0f);
	locationField->setName(locationFieldString);
	addExposedField(locationField);

	// direction exposed field
	directionField = new SFVec3f(0.0f, 0.0f, -1.0f);
	directionField->setName(directionFieldString);
	addExposedField(directionField);

	// radius exposed field
	radiusField = new SFFloat(100.0f);
	radiusField->setName(radiusFieldString);
	addExposedField(radiusField);

	// attenuation exposed field
	attenuationField = new SFVec3f(1.0f, 0.0f, 0.0f);
	attenuationField->setName(attenuationFieldString);
	addExposedField(attenuationField);

	// beamWidth exposed field
	beamWidthField = new SFFloat(1.570796f);
	beamWidthField->setName(beamWidthFieldString);
	addExposedField(beamWidthField);

	// cutOffAngle exposed field
	cutOffAngleField = new SFFloat(0.785398f);
	cutOffAngleField->setName(cutOffAngleFieldString);
	addExposedField(cutOffAngleField);
}

SpotLightNode::~SpotLightNode() 
{
}

////////////////////////////////////////////////
//	Location
////////////////////////////////////////////////

SFVec3f *SpotLightNode::getLocationField() const
{
	if (isInstanceNode() == false)
		return locationField;
	return (SFVec3f *)getExposedField(locationFieldString);
}

void SpotLightNode::setLocation(float value[]) 
{
	getLocationField()->setValue(value);
}

void SpotLightNode::setLocation(float x, float y, float z) 
{
	getLocationField()->setValue(x, y, z);
}

void SpotLightNode::getLocation(float value[]) const
{
	getLocationField()->getValue(value);
}

////////////////////////////////////////////////
//	Direction
////////////////////////////////////////////////

SFVec3f *SpotLightNode::getDirectionField() const
{
	if (isInstanceNode() == false)
		return directionField;
	return (SFVec3f *)getExposedField(directionFieldString);
}

void SpotLightNode::setDirection(float value[]) 
{
	getDirectionField()->setValue(value);
}

void SpotLightNode::setDirection(float x, float y, float z) 
{
	getDirectionField()->setValue(x, y, z);
}

void SpotLightNode::getDirection(float value[]) const
{
	getDirectionField()->getValue(value);
}

////////////////////////////////////////////////
//	Radius
////////////////////////////////////////////////

SFFloat *SpotLightNode::getRadiusField() const
{
	if (isInstanceNode() == false)
		return radiusField;
	return (SFFloat *)getExposedField(radiusFieldString);
}
	
void SpotLightNode::setRadius(float value) 
{
	getRadiusField()->setValue(value);
}

float SpotLightNode::getRadius() const
{
	return getRadiusField()->getValue();
}

////////////////////////////////////////////////
//	Attenuation
////////////////////////////////////////////////

SFVec3f *SpotLightNode::getAttenuationField() const
{
	if (isInstanceNode() == false)
		return attenuationField;
	return (SFVec3f *)getExposedField(attenuationFieldString);
}

void SpotLightNode::setAttenuation(float value[]) 
{
	getAttenuationField()->setValue(value);
}

void SpotLightNode::setAttenuation(float x, float y, float z) 
{
	getAttenuationField()->setValue(x, y, z);
}

void SpotLightNode::getAttenuation(float value[]) const
{
	getAttenuationField()->getValue(value);
}

////////////////////////////////////////////////
//	BeamWidth
////////////////////////////////////////////////

SFFloat *SpotLightNode::getBeamWidthField() const
{
	if (isInstanceNode() == false)
		return beamWidthField;
	return (SFFloat *)getExposedField(beamWidthFieldString);
}
	
void SpotLightNode::setBeamWidth(float value) 
{
	getBeamWidthField()->setValue(value);
}

float SpotLightNode::getBeamWidth() const
{
	return getBeamWidthField()->getValue();
}

////////////////////////////////////////////////
//	CutOffAngle
////////////////////////////////////////////////

SFFloat *SpotLightNode::getCutOffAngleField() const
{
	if (isInstanceNode() == false)
		return cutOffAngleField;
	return (SFFloat *)getExposedField(cutOffAngleFieldString);
}
	
void SpotLightNode::setCutOffAngle(float value) 
{
	getCutOffAngleField()->setValue(value);
}

float SpotLightNode::getCutOffAngle() const
{
	return getCutOffAngleField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

SpotLightNode *SpotLightNode::next() const
{
	return (SpotLightNode *)Node::next(getType());
}

SpotLightNode *SpotLightNode::nextTraversal() const
{
	return (SpotLightNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool SpotLightNode::isChildNodeType(Node *node) const
{
	return false;
}

void SpotLightNode::initialize() 
{
}

void SpotLightNode::uninitialize() 
{
}

void SpotLightNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void SpotLightNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFBool *bon = getOnField();
	SFColor *color = getColorField();
	SFVec3f *direction = getDirectionField();
	SFVec3f *location = getLocationField();
	SFVec3f *attenuation = getAttenuationField();

	printStream << indentString << "\t" << "on " << bon << std::endl;
	printStream << indentString << "\t" << "intensity " << getIntensity() << std::endl;
	printStream << indentString << "\t" << "ambientIntensity " << getAmbientIntensity() << std::endl;
	printStream << indentString << "\t" << "color " << color << std::endl;
	printStream << indentString << "\t" << "direction " << direction << std::endl;
	printStream << indentString << "\t" << "location " << location << std::endl;
	printStream << indentString << "\t" << "beamWidth " << getBeamWidth() << std::endl;
	printStream << indentString << "\t" << "cutOffAngle " << getCutOffAngle() << std::endl;
	printStream << indentString << "\t" << "radius " << getRadius() << std::endl;
	printStream << indentString << "\t" << "attenuation " << attenuation << std::endl;
}
