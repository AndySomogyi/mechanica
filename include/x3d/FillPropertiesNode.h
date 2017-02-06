/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FillPropertiesNode.h
*
******************************************************************/

#ifndef _CX3D_FILLPROPERTIESNODE_H_
#define _CX3D_FILLPROPERTIESNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/AppearanceChildNode.h>

namespace CyberX3D {

class FillPropertiesNode : public AppearanceChildNode {

	SFString *fillStyleField;
	SFInt32 *hatchStyleField;
	SFColor *hatchColorField;
	
public:

	FillPropertiesNode();
	virtual ~FillPropertiesNode();

	////////////////////////////////////////////////
	//	FillStyle
	////////////////////////////////////////////////

	SFString *getFillStyleField() const;
	
	void setFillStyle(const char *value);
	const char *getFillStyle() const;

	////////////////////////////////////////////////
	//	HatchStyle
	////////////////////////////////////////////////

	SFInt32 *getHatchStyleField() const;
	
	void setHatchStyle(int value);
	int getHatchStyle() const;

	////////////////////////////////////////////////
	//	HatchColor
	////////////////////////////////////////////////

	SFColor *getHatchColorField() const;

	void setHatchColor(float value[]);
	void setHatchColor(float r, float g, float b);
	void getHatchColor(float value[]) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	FillPropertiesNode *next() const;
	FillPropertiesNode *nextTraversal() const;

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
