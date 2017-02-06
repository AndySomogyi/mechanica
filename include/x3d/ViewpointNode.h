/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ViewpointNode.h
*
******************************************************************/

#ifndef _CX3D_VIEWPOINT_H_
#define _CX3D_VIEWPOINT_H_

#include <x3d/BindableNode.h>

namespace CyberX3D {

class ViewpointNode : public BindableNode {

	SFVec3f *positionField;
	SFRotation *orientationField;
	SFString *descriptionField;
	SFFloat *fovField;
	SFBool *jumpField;

public:

	ViewpointNode();
	virtual ~ViewpointNode();

	////////////////////////////////////////////////
	//	Jump
	////////////////////////////////////////////////

	SFBool *getJumpField() const;
	
	void setJump(bool value);
	void setJump(int value);
	bool getJump() const;

	////////////////////////////////////////////////
	//	FieldOfView
	////////////////////////////////////////////////

	SFFloat *getFieldOfViewField() const;
	
	void setFieldOfView(float value);
	float getFieldOfView() const;

	////////////////////////////////////////////////
	//	Description
	////////////////////////////////////////////////

	SFString *getDescriptionField() const;
	
	void setDescription(const char *value);
	const char *getDescription() const;

	////////////////////////////////////////////////
	//	Position
	////////////////////////////////////////////////

	SFVec3f *getPositionField() const;

	void setPosition(float value[]);
	void setPosition(float x, float y, float z);
	void getPosition(float value[]) const;

	////////////////////////////////////////////////
	//	Orientation
	////////////////////////////////////////////////

	SFRotation *getOrientationField() const;

	void setOrientation(float value[]);
	void setOrientation(float x, float y, float z, float w);
	void getOrientation(float value[]) const;

	////////////////////////////////////////////////
	//	Add position
	////////////////////////////////////////////////

	void addPosition(float worldTranslation[3]); 
	void addPosition(float worldx, float worldy, float worldz); 
	void addPosition(float localTranslation[3], float frame[3][3]); 
	void addPosition(float x, float y, float z, float frame[3][3]); 

	////////////////////////////////////////////////
	//	Add orientation
	////////////////////////////////////////////////

	void addOrientation(SFRotation *rot);
	void addOrientation(float rotationValue[4]);
	void addOrientation(float x, float y, float z, float rot);

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	ViewpointNode *next() const;
	ViewpointNode *nextTraversal() const;

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

	void outputContext(std::ostream& printStream, const char *indentString) const;

	////////////////////////////////////////////////
	//	Local frame
	////////////////////////////////////////////////

	void getFrame(float frame[3][3]) const;
	void translate(float vector[3]);
	void translate(SFVec3f vec);
	void rotate(float rotation[4]);
	void rotate(SFRotation rot);

	////////////////////////////////////////////////
	//	ViewpointNode Matrix
	////////////////////////////////////////////////

	void getMatrix(SFMatrix *matrix) const;
	void getMatrix(float value[4][4]) const;
	void getTranslationMatrix(SFMatrix *matrix) const;
	void getTranslationMatrix(float value[4][4]) const;
};

}

#endif

