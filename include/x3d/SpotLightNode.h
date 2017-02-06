/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SpotLightNode.h
*
******************************************************************/

#ifndef _CX3D_SPOTLIGHT_H_
#define _CX3D_SPOTLIGHT_H_

#include <x3d/LightNode.h>

namespace CyberX3D {

class SpotLightNode : public LightNode {

	SFVec3f *locationField;
	SFVec3f *directionField;
	SFFloat *radiusField;
	SFVec3f *attenuationField;
	SFFloat *beamWidthField;
	SFFloat *cutOffAngleField;
	
public:

	SpotLightNode();
	virtual ~SpotLightNode();

	////////////////////////////////////////////////
	//	Location
	////////////////////////////////////////////////

	SFVec3f *getLocationField() const;

	void setLocation(float value[]);
	void setLocation(float x, float y, float z);
	void getLocation(float value[]) const;

	////////////////////////////////////////////////
	//	Direction
	////////////////////////////////////////////////

	SFVec3f *getDirectionField() const;

	void setDirection(float value[]);
	void setDirection(float x, float y, float z);
	void getDirection(float value[]) const;

	////////////////////////////////////////////////
	//	Radius
	////////////////////////////////////////////////
	
	SFFloat *getRadiusField() const;

	void setRadius(float value);
	float getRadius() const;

	////////////////////////////////////////////////
	//	Attenuation
	////////////////////////////////////////////////

	SFVec3f *getAttenuationField() const;

	void setAttenuation(float value[]);
	void setAttenuation(float x, float y, float z);
	void getAttenuation(float value[]) const;

	////////////////////////////////////////////////
	//	BeamWidth
	////////////////////////////////////////////////

	SFFloat *getBeamWidthField() const;
	
	void setBeamWidth(float value);
	float getBeamWidth() const;

	////////////////////////////////////////////////
	//	CutOffAngle
	////////////////////////////////////////////////

	SFFloat *getCutOffAngleField() const;
	
	void setCutOffAngle(float value);
	float getCutOffAngle() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	SpotLightNode *next() const;
	SpotLightNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
