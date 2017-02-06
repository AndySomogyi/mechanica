/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BillboardNode.cpp
*
*	Revisions:
*
*	11/19/02
*		- Changed the super class from GroupingNode to BoundedGroupingNode.
*
******************************************************************/

#include <x3d/SceneGraph.h>
#include <x3d/MathUtil.h>

using namespace CyberX3D;

////////////////////////////////////////////////
//	BillboardNode
////////////////////////////////////////////////

BillboardNode::BillboardNode() 
{
	setHeaderFlag(false);
	setType(BILLBOARD_NODE);

	// axisOfRotation exposed field
	axisOfRotationField = new SFVec3f(0.0f, 1.0f, 0.0f);
	addExposedField(axisOfRotationFieldString, axisOfRotationField);
}

BillboardNode::~BillboardNode() 
{
}

////////////////////////////////////////////////
//	axisOfRotation
////////////////////////////////////////////////

SFVec3f *BillboardNode::getAxisOfRotationField() const
{
	if (isInstanceNode() == false)
		return axisOfRotationField;
	return (SFVec3f *)getExposedField(axisOfRotationFieldString);
}

void BillboardNode::setAxisOfRotation(float value[]) 
{
	getAxisOfRotationField()->setValue(value);
}

void BillboardNode::setAxisOfRotation(float x, float y, float z) 
{
	getAxisOfRotationField()->setValue(x, y, z);
}

void BillboardNode::getAxisOfRotation(float value[])  const
{
	getAxisOfRotationField()->getValue(value);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

BillboardNode *BillboardNode::next()  const
{
	return (BillboardNode *)Node::next(getType());
}

BillboardNode *BillboardNode::nextTraversal()  const
{
	return (BillboardNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool BillboardNode::isChildNodeType(Node *node) const
{
	if (node->isCommonNode() || node->isBindableNode() ||node->isInterpolatorNode() || node->isSensorNode() || node->isGroupingNode() || node->isSpecialGroupNode())
		return true;
	else
		return false;
}

void BillboardNode::initialize() 
{
	recomputeBoundingBox();
}

void BillboardNode::uninitialize() 
{
}

void BillboardNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void BillboardNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	const SFVec3f *axisOfRotation = getAxisOfRotationField();
	printStream << indentString << "\t" << "axisOfRotation " << axisOfRotation << std::endl;
}

////////////////////////////////////////////////
//	BillboardNode::getBillboardNodeToViewerVector
////////////////////////////////////////////////

void BillboardNode::getViewerToBillboardVector(float vector[3]) const
{
	SceneGraph *sg = getSceneGraph();
	ViewpointNode *view = sg->getViewpointNode();
	if (view == NULL)
		view = sg->getDefaultViewpointNode();

	float viewPos[3];
	view->getPosition(viewPos);

	float bboardPos[] = {0.0f, 0.0f, 0.0f};
	
	Node *parentNode = getParentNode();
	if (parentNode != NULL) {
		SFMatrix	mx;
		parentNode->getTransformMatrix(&mx);
		mx.multi(bboardPos);
	}

	SFVec3f	resultVector(bboardPos);
	resultVector.sub(viewPos);
	resultVector.normalize();

	resultVector.getValue(vector);
}

////////////////////////////////////////////////
//	BillboardNode::getBillboardNodeToViewerVector
////////////////////////////////////////////////

void BillboardNode::getBillboardToViewerVector(float vector[3]) const
{
	getViewerToBillboardVector(vector);
	VectorInverse(vector);
}

////////////////////////////////////////////////
//	BillboardNode::getAxisOfRotationAndBillboardNodeToViewerPlaneVector
////////////////////////////////////////////////

void BillboardNode::getPlaneVectorOfAxisOfRotationAndBillboardToViewer(float planeVector[3]) const
{
	float	axisOfRotation[3];
	float	bboardToViewerVector[3];

	getAxisOfRotation(axisOfRotation);
	getBillboardToViewerVector(bboardToViewerVector);
	
	GetPlaneVectorFromTwoVectors(axisOfRotation, bboardToViewerVector, planeVector);
}

////////////////////////////////////////////////
//	BillboardNode::getZAxisVectorOnAxisRotationAndBillboardNodeToViewerPlane
////////////////////////////////////////////////

void BillboardNode::getZAxisVectorOnPlaneOfAxisOfRotationAndBillboardToViewer(float zAxisVectorOnPlane[3]) const
{
	float	axisOfRotation[3];
	float	planeVector[3];

	getAxisOfRotation(axisOfRotation);
	getPlaneVectorOfAxisOfRotationAndBillboardToViewer(planeVector);

	GetPlaneVectorFromTwoVectors(axisOfRotation, planeVector, zAxisVectorOnPlane);
}

////////////////////////////////////////////////
//	BillboardNode::getRotationAngleOfZAxis
////////////////////////////////////////////////

float BillboardNode::getRotationAngleOfZAxis() const
{
	float	axisOfRotation[3];
	float	viewer2bboardVector[3];
	float	planeVector[3];
	float	zAxisVector[] = {0.0f, 0.0f, 1.0f};
	float	zAxisVectorOnPlane[3];

	getAxisOfRotation(axisOfRotation);
	getViewerToBillboardVector(viewer2bboardVector);
	
	GetPlaneVectorFromTwoVectors(axisOfRotation, viewer2bboardVector, planeVector);
	GetPlaneVectorFromTwoVectors(axisOfRotation, planeVector, zAxisVectorOnPlane);
	
	return VectorGetAngle(zAxisVector, zAxisVectorOnPlane);
}

////////////////////////////////////////////////
//	BillboardNode::getRotationAngleOfZAxis
////////////////////////////////////////////////

void BillboardNode::getRotationZAxisRotation(float roationValue[4]) const
{
	float		bboard2viewerVector[3];
	float		planeVector[3];
	float		bboardZAxisVector[] = {0.0f, 0.0f, 1.0f};
	float		bboardZAxisRotationAngle;
	float		bboardYAxisVector[] = {0.0f, 1.0f, 0.0f};
	float		bboardYAxisRotationAngle;

	getBillboardToViewerVector(bboard2viewerVector);

	GetPlaneVectorFromTwoVectors(bboardZAxisVector, bboard2viewerVector, planeVector);
	bboardZAxisRotationAngle = VectorGetAngle(bboardZAxisVector, bboard2viewerVector);
	
	SFRotation	zAxisRotation;
	zAxisRotation.setValue(planeVector, bboardZAxisRotationAngle);
	zAxisRotation.multi(bboardYAxisVector);

	SceneGraph		*sg = getSceneGraph();
	ViewpointNode	*view = sg->getViewpointNode();
	float			viewFrame[3][3];

	if (view == NULL)
		view = sg->getDefaultViewpointNode();

	view->getFrame(viewFrame);
	bboardYAxisRotationAngle = VectorGetAngle(viewFrame[1], bboardYAxisVector);
	if (viewFrame[1][0] > 0.0f)
		bboardYAxisRotationAngle = PI*2.0f - bboardYAxisRotationAngle;

	SFRotation	yAxisRotation;
	yAxisRotation.setValue(viewFrame[2], bboardYAxisRotationAngle);

	SFRotation	rotation;
	rotation.add(&yAxisRotation);
	rotation.add(&zAxisRotation);
	rotation.getValue(roationValue);
}

/*
void BillboardNode::getRotationZAxisRotation(float roationValue[4])
{
	SceneGraph *sg = getSceneGraph();
	ViewpointNode *view = sg->getViewpointNode();
	if (view == NULL)
		view = sg->getDefaultViewpointNode();

	SFMatrix mxView;
	view->getMatrix(&mxView);

	SFMatrix mxBillboard;
	Node *parentNode = getParentNode();
	if (parentNode != NULL) 
		parentNode->getTransformMatrix(&mxBillboard);

	mxBillboard.invert();
	mxView.add(&mxBillboard);

	SFRotation	rotation(&mxView);
	rotation.getValue(roationValue);
}
*/

////////////////////////////////////////////////
//	BillboardNode::getSFMatrix
////////////////////////////////////////////////

void BillboardNode::getSFMatrix(SFMatrix *mOut) const
{
	float	rotationValue[4];
	getAxisOfRotation(rotationValue);

	if (VectorGetLength(rotationValue) > 0.0f) {
		rotationValue[3] = -getRotationAngleOfZAxis();
		SFRotation rotation(rotationValue);
		rotation.getSFMatrix(mOut);
	}
	else {
		getRotationZAxisRotation(rotationValue);
		SFRotation rotation(rotationValue);
		rotation.getSFMatrix(mOut);
	}
}
