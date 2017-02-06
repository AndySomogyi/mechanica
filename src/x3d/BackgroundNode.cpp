/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BackgroundNode.cpp
*
******************************************************************/

#include <x3d/BackgroundNode.h>

using namespace CyberX3D;

BackgroundNode::BackgroundNode()
{
	setHeaderFlag(false);
	setType(BACKGROUND_NODE);

	// groundColor exposed field
	groundColorField = new MFColor();
	addExposedField(groundColorFieldString, groundColorField);

	// skyColor exposed field
	skyColorField = new MFColor();
	addExposedField(skyColorFieldString, skyColorField);

	// groundAngle exposed field
	groundAngleField = new MFFloat();
	addExposedField(groundAngleFieldString, groundAngleField);

	// skyAngle exposed field
	skyAngleField = new MFFloat();
	addExposedField(skyAngleFieldString, skyAngleField);

	// url exposed field
	frontUrlField = new MFString();
	addExposedField(frontUrlFieldString, frontUrlField);

	// url exposed field
	backUrlField = new MFString();
	addExposedField(backUrlFieldString, backUrlField);

	// url exposed field
	leftUrlField = new MFString();
	addExposedField(leftUrlFieldString, leftUrlField);

	// url exposed field
	rightUrlField = new MFString();
	addExposedField(rightUrlFieldString, rightUrlField);

	// url exposed field
	topUrlField = new MFString();
	addExposedField(topUrlFieldString, topUrlField);

	// url exposed field
	bottomUrlField = new MFString();
	addExposedField(bottomUrlFieldString, bottomUrlField);
}

BackgroundNode::~BackgroundNode()
{
}

////////////////////////////////////////////////
// groundColor
////////////////////////////////////////////////

MFColor *BackgroundNode::getGroundColorField() const
{
	if (isInstanceNode() == false)
		return groundColorField;
	return (MFColor *)getExposedField(groundColorFieldString);
}

void BackgroundNode::addGroundColor(float value[])
{
	getGroundColorField()->addValue(value);
}

int BackgroundNode::getNGroundColors() const
{
	return getGroundColorField()->getSize();
}

void BackgroundNode::getGroundColor(int index, float value[]) const
{
	getGroundColorField()->get1Value(index, value);
}

////////////////////////////////////////////////
// skyColor
////////////////////////////////////////////////

MFColor *BackgroundNode::getSkyColorField() const
{
	if (isInstanceNode() == false)
		return skyColorField;
	return (MFColor *)getExposedField(skyColorFieldString);
}

void BackgroundNode::addSkyColor(float value[])
{
	getSkyColorField()->addValue(value);
}

int BackgroundNode::getNSkyColors() const
{
	return getSkyColorField()->getSize();
}

void BackgroundNode::getSkyColor(int index, float value[]) const
{
	getSkyColorField()->get1Value(index, value);
}

////////////////////////////////////////////////
// groundAngle
////////////////////////////////////////////////

MFFloat *BackgroundNode::getGroundAngleField() const
{
	if (isInstanceNode() == false)
		return groundAngleField;
	return (MFFloat *)getExposedField(groundAngleFieldString);
}

void BackgroundNode::addGroundAngle(float value)
{
	getGroundAngleField()->addValue(value);
}

int BackgroundNode::getNGroundAngles() const
{
	return getGroundAngleField()->getSize();
}

float BackgroundNode::getGroundAngle(int index) const
{
	return getGroundAngleField()->get1Value(index);
}

////////////////////////////////////////////////
// skyAngle
////////////////////////////////////////////////

MFFloat *BackgroundNode::getSkyAngleField() const
{
	if (isInstanceNode() == false)
		return skyAngleField;
	return (MFFloat *)getExposedField(skyAngleFieldString);
}

void BackgroundNode::addSkyAngle(float value)
{
	getSkyAngleField()->addValue(value);
}

int BackgroundNode::getNSkyAngles() const
{
	return getSkyAngleField()->getSize();
}

float BackgroundNode::getSkyAngle(int index) const
{
	return getSkyAngleField()->get1Value(index);
}

////////////////////////////////////////////////
// frontUrl
////////////////////////////////////////////////

MFString *BackgroundNode::getFrontUrlField() const
{
	if (isInstanceNode() == false)
		return frontUrlField;
	return (MFString *)getExposedField(frontUrlFieldString);
}

void BackgroundNode::addFrontUrl(const char * value)
{
	getFrontUrlField()->addValue(value);
}

int BackgroundNode::getNFrontUrls() const
{
	return getFrontUrlField()->getSize();
}

const char * BackgroundNode::getFrontUrl(int index) const
{
	return getFrontUrlField()->get1Value(index);
}

////////////////////////////////////////////////
// backUrl
////////////////////////////////////////////////

MFString *BackgroundNode::getBackUrlField() const
{
	if (isInstanceNode() == false)
		return backUrlField;
	return (MFString *)getExposedField(backUrlFieldString);
}

void BackgroundNode::addBackUrl(const char * value)
{
	getBackUrlField()->addValue(value);
}

int BackgroundNode::getNBackUrls() const
{
	return getBackUrlField()->getSize();
}

const char * BackgroundNode::getBackUrl(int index) const
{
	return getBackUrlField()->get1Value(index);
}

////////////////////////////////////////////////
// leftUrl
////////////////////////////////////////////////

MFString *BackgroundNode::getLeftUrlField() const
{
	if (isInstanceNode() == false)
		return leftUrlField;
	return (MFString *)getExposedField(leftUrlFieldString);
}

void BackgroundNode::addLeftUrl(const char * value)
{
	getLeftUrlField()->addValue(value);
}

int BackgroundNode::getNLeftUrls() const
{
	return getLeftUrlField()->getSize();
}

const char * BackgroundNode::getLeftUrl(int index) const
{
	return getLeftUrlField()->get1Value(index);
}

////////////////////////////////////////////////
// rightUrl
////////////////////////////////////////////////

MFString *BackgroundNode::getRightUrlField() const
{
	if (isInstanceNode() == false)
		return rightUrlField;
	return (MFString *)getExposedField(rightUrlFieldString);
}

void BackgroundNode::addRightUrl(const char * value) 
{
	getRightUrlField()->addValue(value);
}

int BackgroundNode::getNRightUrls() const
{
	return getRightUrlField()->getSize();
}

const char * BackgroundNode::getRightUrl(int index) const
{
	return getRightUrlField()->get1Value(index);
}

////////////////////////////////////////////////
// topUrl
////////////////////////////////////////////////

MFString *BackgroundNode::getTopUrlField() const
{
	if (isInstanceNode() == false)
		return topUrlField;
	return (MFString *)getExposedField(topUrlFieldString);
}

void BackgroundNode::addTopUrl(const char * value)
{
	getTopUrlField()->addValue(value);
}

int BackgroundNode::getNTopUrls() const
{
	return getTopUrlField()->getSize();
}

const char * BackgroundNode::getTopUrl(int index) const
{
	return getTopUrlField()->get1Value(index);
}

////////////////////////////////////////////////
// bottomUrl
////////////////////////////////////////////////

MFString *BackgroundNode::getBottomUrlField() const
{
	if (isInstanceNode() == false)
		return bottomUrlField;
	return (MFString *)getExposedField(bottomUrlFieldString);
}

void BackgroundNode::addBottomUrl(const char * value)
{
	getBottomUrlField()->addValue(value);
}

int BackgroundNode::getNBottomUrls() const
{
	return getBottomUrlField()->getSize();
}

const char * BackgroundNode::getBottomUrl(int index) const
{
	return getBottomUrlField()->get1Value(index);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

BackgroundNode *BackgroundNode::next() const
{
	return (BackgroundNode *)Node::next(getType());
}

BackgroundNode *BackgroundNode::nextTraversal() const
{
	return (BackgroundNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	Virtual functions
////////////////////////////////////////////////
	
bool BackgroundNode::isChildNodeType(Node *node) const
{
	return false;
}

void BackgroundNode::initialize()
{
}

void BackgroundNode::uninitialize()
{
}

void BackgroundNode::update()
{
}

void BackgroundNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	if (0 < getNGroundColors()){
		const MFColor *groundColor = getGroundColorField();
		printStream << indentString << "\t" << "groundColor [" << std::endl;
		groundColor->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNSkyColors()){
		const MFColor *skyColor = getSkyColorField();
		printStream << indentString << "\t" << "skyColor [" << std::endl;
		skyColor->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNGroundAngles()){
		const MFFloat *groundAngle = getGroundAngleField();
		printStream << indentString << "\t" << "groundAngle [" << std::endl;
		groundAngle->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNSkyAngles()){
		const MFFloat *skyAngle = getSkyAngleField();
		printStream << indentString << "\t" << "skyAngle [" << std::endl;
		skyAngle->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNFrontUrls()){
		const MFString *frontUrl = getFrontUrlField();
		printStream << indentString << "\t" << "frontUrl [" << std::endl;
		frontUrl->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNBackUrls()){
		const MFString *backUrl = getBackUrlField();
		printStream << indentString << "\t" << "backUrl [" << std::endl;
		backUrl->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNLeftUrls()){
		const MFString *leftUrl = getLeftUrlField();
		printStream << indentString << "\t" << "leftUrl [" << std::endl;
		leftUrl->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNRightUrls()){
		const MFString *rightUrl = getRightUrlField();
		printStream << indentString << "\t" << "rightUrl [" << std::endl;
		rightUrl->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNTopUrls()){
		const MFString *topUrl = getTopUrlField();
		printStream << indentString << "\t" << "topUrl [" << std::endl;
		topUrl->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNBottomUrls()){
		const MFString *bottomUrl = getBottomUrlField();
		printStream << indentString << "\t" << "bottomUrl [" << std::endl;
		bottomUrl->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}
