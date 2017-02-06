/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	LinePropertiesNode.h
*
******************************************************************/

#ifndef _CX3D_LINEPROPERTIESNODE_H_
#define _CX3D_LINEPROPERTIESNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/AppearanceChildNode.h>

namespace CyberX3D {

class LinePropertiesNode : public AppearanceChildNode {

	SFInt32 *lineStyleField;
	SFFloat *widthField;
	
public:

	LinePropertiesNode();
	virtual ~LinePropertiesNode();

	////////////////////////////////////////////////
	//	LineStyle
	////////////////////////////////////////////////

	SFInt32 *getLineStyleField() const;
	
	void setLineStyle(int value);
	int getLineStyle() const;

	////////////////////////////////////////////////
	//	Width
	////////////////////////////////////////////////

	SFFloat *getWidthField() const;
	
	void setWidth(float value);
	float getWidth() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	LinePropertiesNode *next() const;
	LinePropertiesNode *nextTraversal() const;

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
