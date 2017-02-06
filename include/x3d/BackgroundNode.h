/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BackgroundNode.h
*
******************************************************************/

#ifndef _CX3D_BACKGROUND_H_
#define _CX3D_BACKGROUND_H_

#include <x3d/BindableNode.h>

namespace CyberX3D {

class BackgroundNode : public BindableNode {

	MFColor *groundColorField;
	MFColor *skyColorField;
	MFFloat *groundAngleField;
	MFFloat *skyAngleField;
	MFString *frontUrlField;
	MFString *backUrlField;
	MFString *leftUrlField;
	MFString *rightUrlField;
	MFString *topUrlField;
	MFString *bottomUrlField;
	
public:

	BackgroundNode();
	virtual ~BackgroundNode();

	////////////////////////////////////////////////
	// groundColor
	////////////////////////////////////////////////

	MFColor *getGroundColorField() const;

	void addGroundColor(float value[]);
	int getNGroundColors() const;
	void getGroundColor(int index, float value[]) const;

	////////////////////////////////////////////////
	// skyColor
	////////////////////////////////////////////////

	MFColor *getSkyColorField() const;

	void addSkyColor(float value[]);
	int getNSkyColors() const;
	void getSkyColor(int index, float value[]) const;

	////////////////////////////////////////////////
	// groundAngle
	////////////////////////////////////////////////

	MFFloat *getGroundAngleField() const;

	void addGroundAngle(float value);
	int getNGroundAngles() const;
	float getGroundAngle(int index) const;

	////////////////////////////////////////////////
	// skyAngle
	////////////////////////////////////////////////

	MFFloat *getSkyAngleField() const;

	void addSkyAngle(float value);
	int getNSkyAngles() const;
	float getSkyAngle(int index) const;

	////////////////////////////////////////////////
	// frontUrl
	////////////////////////////////////////////////

	MFString *getFrontUrlField() const;

	void addFrontUrl(const char *value);
	int getNFrontUrls() const;
	const char *getFrontUrl(int index) const;

	////////////////////////////////////////////////
	// backUrl
	////////////////////////////////////////////////

	MFString *getBackUrlField() const;

	void addBackUrl(const char *value);
	int getNBackUrls() const;
	const char *getBackUrl(int index) const;

	////////////////////////////////////////////////
	// leftUrl
	////////////////////////////////////////////////

	MFString *getLeftUrlField() const;

	void addLeftUrl(const char *value);
	int getNLeftUrls() const;
	const char *getLeftUrl(int index) const;

	////////////////////////////////////////////////
	// rightUrl
	////////////////////////////////////////////////

	MFString *getRightUrlField() const;

	void addRightUrl(const char *value);
	int getNRightUrls() const;
	const char *getRightUrl(int index) const;

	////////////////////////////////////////////////
	// topUrl
	////////////////////////////////////////////////

	MFString *getTopUrlField() const;

	void addTopUrl(const char *value);
	int getNTopUrls() const;
	const char *getTopUrl(int index) const;

	////////////////////////////////////////////////
	// bottomUrl
	////////////////////////////////////////////////

	MFString *getBottomUrlField() const;

	void addBottomUrl(const char *value);
	int getNBottomUrls() const;
	const char *getBottomUrl(int index) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	BackgroundNode *next() const;
	BackgroundNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	Virtual functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();
	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
