/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FogNode.h
*
******************************************************************/

#ifndef _CX3D_FOG_H_
#define _CX3D_FOG_H_

#include <x3d/BindableNode.h>

namespace CyberX3D {

class FogNode : public BindableNode {

	SFColor *colorField;
	SFString *fogTypeField;
	SFFloat *visibilityRangeField;

public:

	FogNode();
	virtual ~FogNode();

	////////////////////////////////////////////////
	//	Color
	////////////////////////////////////////////////

	SFColor *getColorField() const;

	void setColor(float value[]);
	void setColor(float r, float g, float b);
	void getColor(float value[]) const;

	////////////////////////////////////////////////
	//	FogType
	////////////////////////////////////////////////

	SFString *getFogTypeField() const;

	void setFogType(const char *value);
	const char *getFogType() const;

	////////////////////////////////////////////////
	//	VisibilityRange
	////////////////////////////////////////////////

	SFFloat *getVisibilityRangeField() const;

	void setVisibilityRange(float value);
	float getVisibilityRange() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	FogNode *next() const;
	FogNode *nextTraversal() const;

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

