/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	LightNode.h
*
******************************************************************/

#ifndef _CX3D_LIGHTNODE_H_
#define _CX3D_LIGHTNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>

namespace CyberX3D {

class LightNode : public Node {

	SFFloat *ambientIntensityField;
	SFBool *bonField;
	SFFloat *intensityField;
	SFColor *colorField;
	
public:

	LightNode();
	virtual ~LightNode();

	////////////////////////////////////////////////
	//	AmbientIntensity
	////////////////////////////////////////////////
	
	SFFloat *getAmbientIntensityField() const;

	void setAmbientIntensity(float value);
	float getAmbientIntensity() const;

	////////////////////////////////////////////////
	//	On
	////////////////////////////////////////////////

	SFBool *getOnField() const;
	
	void setOn(bool on);
	void setOn(int value);
	bool isOn() const;

	////////////////////////////////////////////////
	//	Intensity
	////////////////////////////////////////////////

	SFFloat *getIntensityField() const;
	
	void setIntensity(float value);
	float getIntensity() const;

	////////////////////////////////////////////////
	//	Color
	////////////////////////////////////////////////

	SFColor *getColorField() const;

	void setColor(float value[]);
	void setColor(float r, float g, float b);
	void getColor(float value[]) const;

	////////////////////////////////////////////////
	//	Diffuse Color
	////////////////////////////////////////////////

	void getDiffuseColor(float value[]) const;

	////////////////////////////////////////////////
	//	Ambient Color
	////////////////////////////////////////////////

	void getAmbientColor(float value[]) const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	virtual bool isChildNodeType(Node *node) const = 0;

	virtual void initialize() = 0;

	virtual void uninitialize() = 0;

	virtual void update() = 0;

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	virtual void outputContext(std::ostream &printStream, const char *indentString) const = 0;
};

}

#endif

