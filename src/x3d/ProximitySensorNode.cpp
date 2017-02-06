/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ProximitySensorNode.cpp
*
*	Revisions:
*
*	12/08/02
*		- Changed the super class from SensorNode to EnvironmentalSensorNode.
*		- Moved the following fields to EnvironmentalSensorNode.
*			center, size, enterTime, exitTime
*
******************************************************************/

#include <x3d/SceneGraph.h>

using namespace CyberX3D;

ProximitySensorNode::ProximitySensorNode() 
{
	setHeaderFlag(false);
	setType(PROXIMITYSENSOR_NODE);

	// position eventOut field
	positionField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addEventOut(positionFieldString, positionField);

	// orientation eventOut field
	orientationField = new SFRotation(0.0f, 0.0f, 1.0f, 0.0f);
	addEventOut(orientationFieldString, orientationField);

	// display list field
	inRegionField = new SFBool(false);
	inRegionField->setName(inRegionPrivateFieldString);
	addPrivateField(inRegionField);
}

ProximitySensorNode::~ProximitySensorNode() 
{
}

////////////////////////////////////////////////
//	Position
////////////////////////////////////////////////

SFVec3f *ProximitySensorNode::getPositionChangedField() const
{
	if (isInstanceNode() == false)
		return positionField;
	return (SFVec3f *)getEventOut(positionFieldString);
}
	
void ProximitySensorNode::setPositionChanged(float value[]) 
{
	getPositionChangedField()->setValue(value);
}

void ProximitySensorNode::setPositionChanged(float x, float y, float z) 
{
	getPositionChangedField()->setValue(x, y, z);
}

void ProximitySensorNode::getPositionChanged(float value[]) const
{
	getPositionChangedField()->getValue(value);
}

////////////////////////////////////////////////
//	Orientation
////////////////////////////////////////////////

SFRotation *ProximitySensorNode::getOrientationChangedField() const
{
	if (isInstanceNode() == false)
		return orientationField;
	return (SFRotation *)getEventOut(orientationFieldString);
}
	
void ProximitySensorNode::setOrientationChanged(float value[]) 
{
	getOrientationChangedField()->setValue(value);
}

void ProximitySensorNode::setOrientationChanged(float x, float y, float z, float rot) 
{
	getOrientationChangedField()->setValue(x, y, z, rot);
}

void ProximitySensorNode::getOrientationChanged(float value[]) const
{
	getOrientationChangedField()->getValue(value);
}

////////////////////////////////////////////////
//	inRegion
////////////////////////////////////////////////

SFBool *ProximitySensorNode::getInRegionField() const
{
	if (isInstanceNode() == false)
		return inRegionField;
	return (SFBool *)getPrivateField(inRegionPrivateFieldString);
}

void ProximitySensorNode::setInRegion(bool value) 
{
	getInRegionField()->setValue(value);
}

bool ProximitySensorNode::inRegion() const
{
	return getInRegionField()->getValue();
} 

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

ProximitySensorNode *ProximitySensorNode::next() const
{
	return (ProximitySensorNode *)Node::next(getType());
}

ProximitySensorNode *ProximitySensorNode::nextTraversal() const
{
	return (ProximitySensorNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool ProximitySensorNode::isChildNodeType(Node *node) const
{
	return false;
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void ProximitySensorNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFBool *enabled = getEnabledField();
	SFVec3f *center = getCenterField();
	SFVec3f *size = getSizeField();

	printStream << indentString << "\t" << "enabled " << enabled << std::endl;
	printStream << indentString << "\t" << "center " << center << std::endl;
	printStream << indentString << "\t" << "size " << size << std::endl;
}

////////////////////////////////////////////////
//	ProximitySensorNode::initialize
////////////////////////////////////////////////

void ProximitySensorNode::initialize() 
{
	setInRegion(false);
}

////////////////////////////////////////////////
//	ProximitySensorNode::uninitialize
////////////////////////////////////////////////

void ProximitySensorNode::uninitialize() 
{
}

////////////////////////////////////////////////
//	ProximitySensorNode::update
////////////////////////////////////////////////

static bool isRegion(float vpos[], float center[], float size[])
{
	for (int n=0; n<3; n++) {
		if (vpos[n] < center[n] - size[n]/2.0f)
			return false;
		if (center[n] + size[n]/2.0f < vpos[n])
			return false;
	}

	return true;
}

void ProximitySensorNode::update() 
{
	if (!isEnabled())
		return;

	SceneGraph *sg = getSceneGraph();
	if (!sg)
		return;

	ViewpointNode *vpoint = sg->getViewpointNode();
	if (vpoint == NULL)
		vpoint = sg->getDefaultViewpointNode();

	float vpos[3];
	vpoint->getPosition(vpos);

	float center[3];
	getCenter(center);

	float size[3];
	getSize(size);

	if (inRegion() == false) {
		if (isRegion(vpos, center, size) == true) {
			setInRegion(true);
			double time = GetCurrentSystemTime();
			setEnterTime(time);
			sendEvent(getEventOut(enterTimeFieldString));
			setIsActive(true);
			sendEvent(getEventOut(isActiveFieldString));
		}
	}
	else {
		if (isRegion(vpos, center, size) == false) {
			setInRegion(false);
			double time = GetCurrentSystemTime();
			setExitTime(time);
			sendEvent(getEventOut(exitTimeFieldString));
			setIsActive(false);
			sendEvent(getEventOut(isActiveFieldString));
		}
	}
}

