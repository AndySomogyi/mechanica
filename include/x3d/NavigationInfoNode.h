/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NavigationInfoNode.h
*
******************************************************************/

#ifndef _CX3D_NAVIGATIONINFO_H_
#define _CX3D_NAVIGATIONINFO_H_

#include <x3d/BindableNode.h>

namespace CyberX3D {

class NavigationInfoNode : public BindableNode {

	SFFloat *visibilityLimitField;
	MFFloat *avatarSizeField;
	MFString *typeField;
	SFBool *headlightField;
	SFFloat *speedField;
	
public:

	NavigationInfoNode();
	virtual ~NavigationInfoNode();

	////////////////////////////////////////////////
	// Type
	////////////////////////////////////////////////

	MFString *getTypeField() const;

	void addType(const char *value);
	int getNTypes() const;
	const char *getType(int index) const;

	////////////////////////////////////////////////
	// avatarSize
	////////////////////////////////////////////////

	MFFloat *getAvatarSizeField() const;

	void addAvatarSize(float value);
	int getNAvatarSizes() const;
	float getAvatarSize(int index) const;

	////////////////////////////////////////////////
	//	Headlight
	////////////////////////////////////////////////

	SFBool *getHeadlightField() const;
	
	void setHeadlight(bool value);
	void setHeadlight(int value);
	bool getHeadlight() const;

	////////////////////////////////////////////////
	//	VisibilityLimit
	////////////////////////////////////////////////

	SFFloat *getVisibilityLimitField() const;

	void setVisibilityLimit(float value);
	float getVisibilityLimit() const;

	////////////////////////////////////////////////
	//	Speed
	////////////////////////////////////////////////
	
	SFFloat *getSpeedField() const;

	void setSpeed(float value);
	float getSpeed() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	bool isChildNodeType(Node *node) const;
	NavigationInfoNode *next() const;
	NavigationInfoNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

