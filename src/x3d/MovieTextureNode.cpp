/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MovieTextureNode.cpp
*
*	Revisions:
*
*	12/05/02
*		- Changed the super class from TextureNode to Texture2DNode.
*
******************************************************************/

#include <x3d/MovieTextureNode.h>

using namespace CyberX3D;

MovieTextureNode::MovieTextureNode() 
{
	setHeaderFlag(false);
	setType(MOVIETEXTURE_NODE);

	///////////////////////////
	// Exposed Field 
	///////////////////////////

	// url field
	urlField = new MFString();
	addExposedField(urlFieldString, urlField);

	// loop exposed field
	loopField = new SFBool(false);
	addExposedField(loopFieldString, loopField);

	// startTime exposed field
	startTimeField = new SFTime(0.0f);
	addExposedField(startTimeFieldString, startTimeField);

	// stopTime exposed field
	stopTimeField = new SFTime(0.0f);
	addExposedField(stopTimeFieldString, stopTimeField);

	// speed exposed field
	speedField = new SFFloat(1.0f);
	addExposedField(speedTimeFieldString, speedField);

	///////////////////////////
	// EventOut
	///////////////////////////

	// isActive eventOut field
	isActiveField = new SFBool(false);
	addEventOut(isActiveFieldString, isActiveField);

	// time eventOut field
	durationChangedField = new SFTime(-1.0f);
	addEventOut(durationFieldString, durationChangedField);
}

MovieTextureNode::~MovieTextureNode() 
{
}

////////////////////////////////////////////////
// Url
////////////////////////////////////////////////

MFString *MovieTextureNode::getUrlField() const
{
	if (isInstanceNode() == false)
		return urlField;
	return (MFString *)getExposedField(urlFieldString);
}

void MovieTextureNode::addUrl(const char *value) 
{
	getUrlField()->addValue(value);
}

int MovieTextureNode::getNUrls() const
{
	return getUrlField()->getSize();
}

const char *MovieTextureNode::getUrl(int index) const
{
	return getUrlField()->get1Value(index);
}

void MovieTextureNode::setUrl(int index, const char *urlString) 
{
	getUrlField()->set1Value(index, urlString);
}

////////////////////////////////////////////////
//	Loop
////////////////////////////////////////////////

SFBool *MovieTextureNode::getLoopField() const
{
	if (isInstanceNode() == false)
		return loopField;
	return (SFBool *)getExposedField(loopFieldString);
}
	
void MovieTextureNode::setLoop(bool value) 
{
	getLoopField()->setValue(value);
}

void MovieTextureNode::setLoop(int value) 
{
	setLoop(value ? true : false);
}

bool MovieTextureNode::getLoop() const
{
	return getLoopField()->getValue();
}

bool MovieTextureNode::isLoop() const
{
	return getLoop();
}

////////////////////////////////////////////////
//	Speed
////////////////////////////////////////////////

SFFloat *MovieTextureNode::getSpeedField() const
{
	if (isInstanceNode() == false)
		return speedField;
	return (SFFloat *)getExposedField(speedTimeFieldString);
}
	
void MovieTextureNode::setSpeed(float value) 
{
	getSpeedField()->setValue(value);
}

float MovieTextureNode::getSpeed() const
{
	return getSpeedField()->getValue();
}

////////////////////////////////////////////////
//	Start time
////////////////////////////////////////////////

SFTime *MovieTextureNode::getStartTimeField() const
{
	if (isInstanceNode() == false)
		return startTimeField;
	return (SFTime *)getExposedField(startTimeFieldString);
}
	
void MovieTextureNode::setStartTime(double value) 
{
	getStartTimeField()->setValue(value);
}

double MovieTextureNode::getStartTime() const
{
	return getStartTimeField()->getValue();
}

////////////////////////////////////////////////
//	Stop time
////////////////////////////////////////////////

SFTime *MovieTextureNode::getStopTimeField() const
{
	if (isInstanceNode() == false)
		return stopTimeField;
	return (SFTime *)getExposedField(stopTimeFieldString);
}
	
void MovieTextureNode::setStopTime(double value) 
{
	getStopTimeField()->setValue(value);
}

double MovieTextureNode::getStopTime() const
{
	return getStopTimeField()->getValue();
}

////////////////////////////////////////////////
//	isActive
////////////////////////////////////////////////

SFBool *MovieTextureNode::getIsActiveField() const
{
	if (isInstanceNode() == false)
		return isActiveField;
	return (SFBool *)getEventOut(isActiveFieldString);
}
	
void MovieTextureNode::setIsActive(bool value) 
{
	getIsActiveField()->setValue(value);
}

bool MovieTextureNode::getIsActive() const
{
	return getIsActiveField()->getValue();
}

bool MovieTextureNode::isActive() const
{
	return getIsActiveField()->getValue();
}

////////////////////////////////////////////////
//	duration_changed
////////////////////////////////////////////////

SFTime *MovieTextureNode::getDurationChangedField() const
{
	if (isInstanceNode() == false)
		return durationChangedField;
	return (SFTime *)getEventOut(durationFieldString);
}
	
void MovieTextureNode::setDurationChanged(double value) 
{
	getDurationChangedField()->setValue(value);
}

double MovieTextureNode::getDurationChanged() const
{
	return getDurationChangedField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

MovieTextureNode *MovieTextureNode::next() const
{
	return (MovieTextureNode *)Node::next(getType());
}

MovieTextureNode *MovieTextureNode::nextTraversal() const
{
	return (MovieTextureNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool MovieTextureNode::isChildNodeType(Node *node) const
{
	return false;
}

void MovieTextureNode::initialize() 
{
}

void MovieTextureNode::uninitialize() 
{
}

void MovieTextureNode::update() 
{
}

////////////////////////////////////////////////
//	infomation
////////////////////////////////////////////////

void MovieTextureNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	const SFBool *loop = getLoopField();
	const SFBool *repeatS = getRepeatSField();
	const SFBool *repeatT = getRepeatTField();

	printStream << indentString << "\t" << "loop " << loop << std::endl;
	printStream << indentString << "\t" << "speed " << getSpeed() << std::endl;
	printStream << indentString << "\t" << "startTime " << getStartTime() << std::endl;
	printStream << indentString << "\t" << "stopTime " << getStopTime() << std::endl;
	printStream << indentString << "\t" << "repeatS " << repeatS << std::endl;
	printStream << indentString << "\t" << "repeatT " << repeatT << std::endl;

	if (0 < getNUrls()) {
		const MFString *url = getUrlField();
		printStream << indentString << "\t" << "url [" << std::endl;
		url->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}
