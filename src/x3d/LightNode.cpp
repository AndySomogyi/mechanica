/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	LightNode.cpp
*
*	Revisions:
*
*	12/05/02
*		- Added a ambientIntensity field.
*		- Added getDiffuseColor(), getAmbientColor()
*
******************************************************************/

#include <x3d/LightNode.h>

using namespace CyberX3D;

LightNode::LightNode() 
{
	setHeaderFlag(false);

	// ambient intensity exposed field
	ambientIntensityField = new SFFloat(0.0f);
	ambientIntensityField->setName(ambientIntensityFieldString);
	addExposedField(ambientIntensityField);

	// on exposed field
	bonField = new SFBool(true);
	bonField->setName(onFieldString);
	addExposedField(bonField);

	// intensity exposed field
	intensityField = new SFFloat(1.0f);
	intensityField->setName(intensityFieldString);
	addExposedField(intensityField);

	// color exposed field
	colorField = new SFColor(1.0f, 1.0f, 1.0f);
	colorField->setName(colorFieldString);
	addExposedField(colorField);
}

LightNode::~LightNode() 
{
}

////////////////////////////////////////////////
//	AmbientIntensity
////////////////////////////////////////////////

SFFloat *LightNode::getAmbientIntensityField() const
{
	if (isInstanceNode() == false)
		return ambientIntensityField;
	return (SFFloat *)getExposedField(ambientIntensityFieldString);
}
	
void LightNode::setAmbientIntensity(float value) 
{
	getAmbientIntensityField()->setValue(value);
}

float LightNode::getAmbientIntensity() const
{
	return getAmbientIntensityField()->getValue();
}

////////////////////////////////////////////////
//	On
////////////////////////////////////////////////

SFBool *LightNode::getOnField() const
{
	if (isInstanceNode() == false)
		return bonField;
	return (SFBool *)getExposedField(onFieldString);
}
	
void LightNode::setOn(bool on) 
{
	getOnField()->setValue(on);
}

void LightNode::setOn(int value) 
{
	setOn(value ? true : false);
}

bool LightNode::isOn() const
{
	return getOnField()->getValue();
}

////////////////////////////////////////////////
//	Intensity
////////////////////////////////////////////////

SFFloat *LightNode::getIntensityField() const
{
	if (isInstanceNode() == false)
		return intensityField;
	return (SFFloat *)getExposedField(intensityFieldString);
}
	
void LightNode::setIntensity(float value) 
{
	getIntensityField()->setValue(value);
}

float LightNode::getIntensity() const
{
	return getIntensityField()->getValue();
}

////////////////////////////////////////////////
//	Color
////////////////////////////////////////////////

SFColor *LightNode::getColorField() const
{
	if (isInstanceNode() == false)
		return colorField;
	return (SFColor *)getExposedField(colorFieldString);
}

void LightNode::setColor(float value[])
{
	getColorField()->setValue(value);
}

void LightNode::setColor(float r, float g, float b) 
{
	getColorField()->setValue(r, g, b);
}

void LightNode::getColor(float value[]) const
{
	getColorField()->getValue(value);
}

////////////////////////////////////////////////
//	Diffuse Color
////////////////////////////////////////////////

void LightNode::getDiffuseColor(float value[])  const
{
	getColor(value);
	float	intensity = getIntensity();
	value[0] *= intensity;
	value[1] *= intensity;
	value[2] *= intensity;
}

////////////////////////////////////////////////
//	Ambient Color
////////////////////////////////////////////////

void LightNode::getAmbientColor(float value[]) const
{
	getColor(value);
	float	intensity = getIntensity();
	float	ambientIntensity = getAmbientIntensity();
	value[0] *= intensity * ambientIntensity;
	value[1] *= intensity * ambientIntensity;
	value[2] *= intensity * ambientIntensity;
}
