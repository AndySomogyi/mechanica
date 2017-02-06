/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PointLightNode.h
*
******************************************************************/

#ifndef _CX3D_POINTLIGHTNODE_H_
#define _CX3D_POINTLIGHTNODE_H_

#include <x3d/LightNode.h>

namespace CyberX3D {

class PointLightNode : public LightNode {

	SFVec3f *locationField;
	SFFloat *radiusField;
	SFVec3f *attenuationField;

public:

	PointLightNode();
	virtual ~PointLightNode();

	////////////////////////////////////////////////
	//	Location
	////////////////////////////////////////////////

	SFVec3f *getLocationField() const;

	void setLocation(float value[]);
	void setLocation(float x, float y, float z);
	void getLocation(float value[]) const;

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
	//	List
	////////////////////////////////////////////////

	PointLightNode *next() const;
	PointLightNode *nextTraversal() const;

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
