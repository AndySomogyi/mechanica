/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ViewpointNode.h
*
******************************************************************/

#include <x3d/ViewpointNode.h>

using namespace CyberX3D;

ViewpointNode::ViewpointNode() 
{
	setHeaderFlag(false);
	setType(VIEWPOINT_NODE);

	// position exposed field
	positionField = new SFVec3f(0.0f, 0.0f, 0.0f);
	positionField->setName(positionFieldString);
	addExposedField(positionField);

	// orientation exposed field
	orientationField = new SFRotation(0.0f, 0.0f, 1.0f, 0.0f);
	orientationField->setName(orientationFieldString);
	addExposedField(orientationField);

	// description exposed field
	descriptionField = new SFString("");
	descriptionField->setName(descriptionFieldString);
	addField(descriptionField);

	// fov exposed field
	fovField = new SFFloat(0.785398f);
	fovField->setName(fieldOfViewFieldString);
	addExposedField(fovField);

	// jump exposed field
	jumpField = new SFBool(true);
	jumpField->setName(jumpFieldString);
	addExposedField(jumpField);
}

ViewpointNode::~ViewpointNode() 
{
}

////////////////////////////////////////////////
//	Jump
////////////////////////////////////////////////

SFBool *ViewpointNode::getJumpField() const
{
	if (isInstanceNode() == false)
		return jumpField;
	return (SFBool *)getExposedField(jumpFieldString);
}
	
void ViewpointNode::setJump(bool value) 
{
	getJumpField()->setValue(value);
}

void ViewpointNode::setJump(int value) 
{
	setJump(value ? true : false);
}

bool ViewpointNode::getJump() const
{
	return getJumpField()->getValue();
}

////////////////////////////////////////////////
//	FieldOfView
////////////////////////////////////////////////

SFFloat *ViewpointNode::getFieldOfViewField() const
{
	if (isInstanceNode() == false)
		return fovField;
	return (SFFloat *)getExposedField(fieldOfViewFieldString);
}
	
void ViewpointNode::setFieldOfView(float value) 
{
	getFieldOfViewField()->setValue(value);
}

float ViewpointNode::getFieldOfView() const
{
	return getFieldOfViewField()->getValue();
}

////////////////////////////////////////////////
//	Description
////////////////////////////////////////////////

SFString *ViewpointNode::getDescriptionField() const
{
	if (isInstanceNode() == false)
		return descriptionField;
	return (SFString *)getField(descriptionFieldString);
}
	
void ViewpointNode::setDescription(const char *value)  
{
	getDescriptionField()->setValue(value);
}

const char *ViewpointNode::getDescription() const
{
	return getDescriptionField()->getValue();
}
////////////////////////////////////////////////
//	Position
////////////////////////////////////////////////

SFVec3f *ViewpointNode::getPositionField() const
{
	if (isInstanceNode() == false)
		return positionField;
	return (SFVec3f *)getExposedField(positionFieldString);
}

void ViewpointNode::setPosition(float value[]) 
{
	getPositionField()->setValue(value);
}

void ViewpointNode::setPosition(float x, float y, float z) 
{
	getPositionField()->setValue(x, y, z);
}

void ViewpointNode::getPosition(float value[]) const
{
	getPositionField()->getValue(value);
}

////////////////////////////////////////////////
//	Orientation
////////////////////////////////////////////////

SFRotation *ViewpointNode::getOrientationField() const
{
	if (isInstanceNode() == false)
		return orientationField;
	return (SFRotation *)getExposedField(orientationFieldString);
}

void ViewpointNode::setOrientation(float value[]) 
{
	getOrientationField()->setValue(value);
}

void ViewpointNode::setOrientation(float x, float y, float z, float w) 
{
	getOrientationField()->setValue(x, y, z, w);
}

void ViewpointNode::getOrientation(float value[]) const
{
	getOrientationField()->getValue(value);
}

////////////////////////////////////////////////
//	Add position
////////////////////////////////////////////////

void ViewpointNode::addPosition(float worldTranslation[3]) 
{ 
	getPositionField()->add(worldTranslation);
}

void ViewpointNode::addPosition(float worldx, float worldy, float worldz) 
{ 
	getPositionField()->add(worldx, worldy, worldz);
}

void ViewpointNode::addPosition(float localTranslation[3], float frame[3][3]) 
{ 
	SFVec3f *position = getPositionField();
	float	translation[3];
	for (int axis=0; axis<3; axis++) {
		SFVec3f vector(frame[axis]);
		vector.scale(localTranslation[axis]);
		vector.getValue(translation);
		position->add(translation);
	}
}

void ViewpointNode::addPosition(float x, float y, float z, float frame[3][3]) 
{ 
	float localTranslation[3];
	localTranslation[0] = x;
	localTranslation[1] = y;
	localTranslation[2] = z;
	addPosition(localTranslation, frame);
}

////////////////////////////////////////////////
//	Add orientation
////////////////////////////////////////////////

void ViewpointNode::addOrientation(SFRotation *rot) 
{
	getOrientationField()->add(rot);
}

void ViewpointNode::addOrientation(float rotationValue[4]) 
{
	getOrientationField()->add(rotationValue);
}

void ViewpointNode::addOrientation(float x, float y, float z, float rot) 
{
	getOrientationField()->add(x, y, z, rot);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

ViewpointNode *ViewpointNode::next() const
{
	return (ViewpointNode *)Node::next(getType());
}

ViewpointNode *ViewpointNode::nextTraversal() const
{
	return (ViewpointNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool ViewpointNode::isChildNodeType(Node *node) const
{
	return false;
}

void ViewpointNode::initialize() 
{
}

void ViewpointNode::uninitialize() 
{
}

void ViewpointNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void ViewpointNode::outputContext(std::ostream& printStream, const char *indentString) const
{
	SFVec3f *position = getPositionField();
	SFRotation *orientation = getOrientationField();
	SFBool *jump = getJumpField();
	SFString *description = getDescriptionField();
	printStream << indentString << "\t" << "fieldOfView " << getFieldOfView() << std::endl;
	printStream << indentString << "\t" << "jump " << jump << std::endl;
	printStream << indentString << "\t" << "position " << position << std::endl;
	printStream << indentString << "\t" << "orientation " << orientation << std::endl;
	printStream << indentString << "\t" << "description " << description << std::endl;
}

////////////////////////////////////////////////
//	Local frame
////////////////////////////////////////////////

void ViewpointNode::getFrame(float frame[3][3]) const
{
	SFRotation *orientation = getOrientationField();
	// local x frame
	frame[0][0] = 1.0f;
	frame[0][1] = 0.0f;
	frame[0][2] = 0.0f;
	orientation->multi(frame[0]);
	// local 0 frame
	frame[1][0] = 0.0f;
	frame[1][1] = 1.0f;
	frame[1][2] = 0.0f;
	orientation->multi(frame[1]);
	// local 0 frame
	frame[2][0] = 0.0f;
	frame[2][1] = 0.0f;
	frame[2][2] = 1.0f;
	orientation->multi(frame[2]);
}

void ViewpointNode::translate(float vector[3]) 
{
	float frame[3][3];
	getFrame(frame);
	addPosition(vector, frame);
}

void ViewpointNode::translate(SFVec3f vec) 
{
	float frame[3][3];
	float vector[3];
	getFrame(frame);
	vec.getValue(vector);
	addPosition(vector, frame);
}

void ViewpointNode::rotate(float rotation[4]) 
{
	addOrientation(rotation);
}

void ViewpointNode::rotate(SFRotation rot) 
{
	float rotation[4];
	rot.getValue(rotation);
	addOrientation(rotation);
}

////////////////////////////////////////////////
//	ViewpointNode Matrix
////////////////////////////////////////////////

void ViewpointNode::getMatrix(SFMatrix *matrix) const
{
	float	position[3];
	float	rotation[4];
	
	getPosition(position);
	SFVec3f	transView(position);
	transView.invert();

	getOrientation(rotation);
	SFRotation rotView(rotation);
	rotView.invert();

	SFMatrix	mxTrans, mxRot;
	mxTrans.setTranslation(&transView);
	mxRot.setRotation(&rotView);

	matrix->init();
	matrix->add(&mxRot);
	matrix->add(&mxTrans);
}

void ViewpointNode::getMatrix(float value[4][4]) const
{
	SFMatrix	mx;
	getMatrix(&mx);
	mx.getValue(value);
}

void ViewpointNode::getTranslationMatrix(SFMatrix *matrix) const
{
	float	position[3];
	
	getPosition(position);
	SFVec3f	transView(position);
	transView.invert();

	SFMatrix	mxTrans;
	mxTrans.setTranslation(&transView);

	matrix->init();
	matrix->add(&mxTrans);
}

void ViewpointNode::getTranslationMatrix(float value[4][4]) const
{
	SFMatrix	mx;
	getTranslationMatrix(&mx);
	mx.getValue(value);
}
