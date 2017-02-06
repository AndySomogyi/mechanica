/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MaterialNode.h
*
******************************************************************/

#ifndef _CX3D_MATERIALNODE_H_
#define _CX3D_MATERIALNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/AppearanceChildNode.h>

namespace CyberX3D {

class MaterialNode : public AppearanceChildNode {

	SFFloat *transparencyField;
	SFFloat *ambientIntensityField;
	SFFloat *shininessField;
	SFColor *diffuseColorField;
	SFColor *specularColorField;
	SFColor *emissiveColorField;
	
public:

	MaterialNode();
	virtual ~MaterialNode();

	////////////////////////////////////////////////
	//	Transparency
	////////////////////////////////////////////////

	SFFloat *getTransparencyField() const;
	
	void setTransparency(float value);
	float getTransparency() const;

	////////////////////////////////////////////////
	//	AmbientIntensity
	////////////////////////////////////////////////

	SFFloat *getAmbientIntensityField() const;
	
	void setAmbientIntensity(float intensity);
	float getAmbientIntensity() const;

	////////////////////////////////////////////////
	//	Shininess
	////////////////////////////////////////////////

	SFFloat *getShininessField() const;
	
	void setShininess(float value);
	float getShininess() const;

	////////////////////////////////////////////////
	//	DiffuseColor
	////////////////////////////////////////////////

	SFColor *getDiffuseColorField() const;

	void setDiffuseColor(float value[]);
	void setDiffuseColor(float r, float g, float b);
	void getDiffuseColor(float value[]) const;

	////////////////////////////////////////////////
	//	SpecularColor
	////////////////////////////////////////////////

	SFColor *getSpecularColorField() const;

	void setSpecularColor(float value[]);
	void setSpecularColor(float r, float g, float b);
	void getSpecularColor(float value[]) const;

	////////////////////////////////////////////////
	//	EmissiveColor
	////////////////////////////////////////////////

	SFColor *getEmissiveColorField() const;

	void setEmissiveColor(float value[]);
	void setEmissiveColor(float r, float g, float b);
	void getEmissiveColor(float value[]) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	MaterialNode *next() const;
	MaterialNode *nextTraversal() const;

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
