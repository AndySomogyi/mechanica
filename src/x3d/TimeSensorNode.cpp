/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TimeSensorNode.cpp
*
*	Revisions:
*		11/13/02
*			- Added the following fields
*				isPaused, pauseTime, resumeTime, elapsedTime, numLoops
*
******************************************************************/

#include <x3d/TimeSensorNode.h>

using namespace CyberX3D;

static const char isPausedFieldName[] = "isPaused";
static const char pauseTimeFieldName[] = "pauseTime";
static const char resumeTimeFieldName[] = "resumeTime";
static const char elapsedTimeFieldName[] = "elapsedTime";
static const char numLoopsFieldName[] = "numLoops";

TimeSensorNode::TimeSensorNode() 
{
	setHeaderFlag(false);
	setType(TIMESENSOR_NODE);

	///////////////////////////////////////
	// VRML97 Fields
	///////////////////////////////////////

	// loop exposed field
	loopField = new SFBool(false);
	addExposedField(loopFieldString, loopField);

	// cybleInterval exposed field
	cycleIntervalField = new SFTime(1.0);
	addExposedField(cycleIntervalFieldString, cycleIntervalField);

	// startTime exposed field
	startTimeField = new SFTime(0.0f);
	addExposedField(startTimeFieldString, startTimeField);

	// stopTime exposed field
	stopTimeField = new SFTime(0.0f);
	addExposedField(stopTimeFieldString, stopTimeField);

	// cycleTime eventOut field
	cycleTimeField = new SFTime(-1.0f);
	addEventOut(cycleTimeFieldString, cycleTimeField);

	// time eventOut field
	timeField = new SFTime(-1.0f);
	addEventOut(timeFieldString, timeField);

	// fraction_changed eventOut field
	fractionChangedField = new SFFloat(0.0f);
	addEventOut(fractionFieldString, fractionChangedField);

	///////////////////////////////////////
	// X3D Fields
	///////////////////////////////////////
		
	// isPaused eventOut field
	isPausedField = new SFBool(false);
	addEventOut(isPausedFieldName, isPausedField);

	// pauseTime exposed field
	pauseTimeField = new SFTime(0.0);
	addExposedField(pauseTimeFieldName, pauseTimeField);

	// resumeTime exposed field
	resumeTimeField = new SFTime(0.0);
	addExposedField(resumeTimeFieldName, resumeTimeField);

	// elapsedTime eventOut field
	elapsedTimeField = new SFTime(0.0);
	addEventOut(elapsedTimeFieldName, elapsedTimeField);

	// numLoops exposed field
	numLoopsField = new SFFloat(1.0f);
	addExposedField(numLoopsFieldName, numLoopsField);
}

TimeSensorNode::~TimeSensorNode() 
{
}

////////////////////////////////////////////////
//	Loop
////////////////////////////////////////////////

SFBool *TimeSensorNode::getLoopField() const
{
	if (isInstanceNode() == false)
		return loopField;
	return (SFBool *)getExposedField(loopFieldString);
}
	
void TimeSensorNode::setLoop(bool value) 
{
	getLoopField()->setValue(value);
}

void TimeSensorNode::setLoop(int value) 
{
	setLoop(value ? true : false);
}

bool TimeSensorNode::getLoop() const
{
	return getLoopField()->getValue();
}

bool TimeSensorNode::isLoop() const
{
	return getLoop();
}

////////////////////////////////////////////////
//	Cyble Interval
////////////////////////////////////////////////

SFTime *TimeSensorNode::getCycleIntervalField() const
{
	if (isInstanceNode() == false)
		return cycleIntervalField;
	return (SFTime *)getExposedField(cycleIntervalFieldString);
}
	
void TimeSensorNode::setCycleInterval(double value) 
{
	getCycleIntervalField()->setValue(value);
}

double TimeSensorNode::getCycleInterval() const
{
	return getCycleIntervalField()->getValue();
}

////////////////////////////////////////////////
//	Start time
////////////////////////////////////////////////

SFTime *TimeSensorNode::getStartTimeField() const
{
	if (isInstanceNode() == false)
		return startTimeField;
	return (SFTime *)getExposedField(startTimeFieldString);
}
	
void TimeSensorNode::setStartTime(double value) 
{
	getStartTimeField()->setValue(value);
}

double TimeSensorNode::getStartTime() const
{
	return getStartTimeField()->getValue();
}

////////////////////////////////////////////////
//	Stop time
////////////////////////////////////////////////

SFTime *TimeSensorNode::getStopTimeField() const
{
	if (isInstanceNode() == false)
		return stopTimeField;
	return (SFTime *)getExposedField(stopTimeFieldString);
}
	
void TimeSensorNode::setStopTime(double value) 
{
	getStopTimeField()->setValue(value);
}

double TimeSensorNode::getStopTime() const
{
	return getStopTimeField()->getValue();
}

////////////////////////////////////////////////
//	fraction_changed
////////////////////////////////////////////////

SFFloat *TimeSensorNode::getFractionChangedField() const
{
	if (isInstanceNode() == false)
		return fractionChangedField;
	return (SFFloat *)getEventOut(fractionFieldString);
}
	
void TimeSensorNode::setFractionChanged(float value) 
{
	getFractionChangedField()->setValue(value);
}

float TimeSensorNode::getFractionChanged() const
{
	return getFractionChangedField()->getValue();
}

////////////////////////////////////////////////
//	Cycle time
////////////////////////////////////////////////

SFTime *TimeSensorNode::getCycleTimeField() const
{
	if (isInstanceNode() == false)
		return cycleTimeField;
	return (SFTime *)getEventOut(cycleTimeFieldString);
}
	
void TimeSensorNode::setCycleTime(double value) 
{
	getCycleTimeField()->setValue(value);
}

double TimeSensorNode::getCycleTime() const
{
	return getCycleTimeField()->getValue();
}

////////////////////////////////////////////////
//	Time
////////////////////////////////////////////////

SFTime *TimeSensorNode::getTimeField() const
{
	if (isInstanceNode() == false)
		return timeField;
	return (SFTime *)getEventOut(timeFieldString);
}
	
void TimeSensorNode::setTime(double value)
{
	getTimeField()->setValue(value);
}

double TimeSensorNode::getTime() const
{
	return getTimeField()->getValue();
}

////////////////////////////////////////////////
//	IsPaused (X3D)
////////////////////////////////////////////////

SFBool *TimeSensorNode::getIsPausedField() const
{
	if (isInstanceNode() == false)
		return isPausedField;
	return (SFBool *)getEventOut(isPausedFieldName);
}
	
void TimeSensorNode::setIsPaused(bool value) 
{
	getIsPausedField()->setValue(value);
}

void TimeSensorNode::setIsPaused(int value) 
{
	setIsPaused(value ? true : false);
}

bool TimeSensorNode::getIsPaused() const
{
	return getIsPausedField()->getValue();
}
	
bool TimeSensorNode::isPaused() const
{
	return getIsPaused();
}

////////////////////////////////////////////////
//	Elapsed time (X3D)
////////////////////////////////////////////////

SFTime *TimeSensorNode::getElapsedTimeField() const
{
	if (isInstanceNode() == false)
		return elapsedTimeField;
	return (SFTime *)getEventOut(elapsedTimeFieldName);
}
	
void TimeSensorNode::setElapsedTime(double value) 
{
	getElapsedTimeField()->setValue(value);
}

double TimeSensorNode::getElapsedTime() const
{
	return getElapsedTimeField()->getValue();
}

////////////////////////////////////////////////
//	Pause time (X3D)
////////////////////////////////////////////////

SFTime *TimeSensorNode::getPauseTimeField() const
{
	if (isInstanceNode() == false)
		return pauseTimeField;
	return (SFTime *)getExposedField(pauseTimeFieldName);
}
	
void TimeSensorNode::setPauseTime(double value) 
{
		getPauseTimeField()->setValue(value);
}

double TimeSensorNode::getPauseTime() const
{
	return getPauseTimeField()->getValue();
}

////////////////////////////////////////////////
//	Resume time (X3D)
////////////////////////////////////////////////

SFTime *TimeSensorNode::getResumeTimeField() const
{
	if (isInstanceNode() == false)
		return resumeTimeField;
	return (SFTime *)getExposedField(resumeTimeFieldName);
}
	
void TimeSensorNode::setResumeTime(double value) 
{
	getResumeTimeField()->setValue(value);
}

double TimeSensorNode::getResumeTime() const
{
	return getResumeTimeField()->getValue();
}

////////////////////////////////////////////////
//	numLoops (X3D)
////////////////////////////////////////////////

SFFloat *TimeSensorNode::getNumLoopsField() const
{
	if (isInstanceNode() == false)
		return numLoopsField;
	return (SFFloat *)getExposedField(numLoopsFieldName);
}
	
void TimeSensorNode::setNumLoops(float value) 
{
	getNumLoopsField()->setValue(value);
}

float TimeSensorNode::getNumLoops() const
{
	return getNumLoopsField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

TimeSensorNode *TimeSensorNode::next() const
{
	return (TimeSensorNode *)Node::next(getType());
}

TimeSensorNode *TimeSensorNode::nextTraversal() const
{
	return (TimeSensorNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	Virtual functions
////////////////////////////////////////////////
	
bool TimeSensorNode::isChildNodeType(Node *node) const
{
	return false;
}

void TimeSensorNode::initialize() 
{
	setCycleTime(-1.0f);
	setIsActive(false);
}

void TimeSensorNode::uninitialize() 
{
}

void TimeSensorNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFBool *bEnabled = getEnabledField();
	SFBool *loop = getLoopField();

	printStream << indentString << "\t" << "cycleInterval " << getCycleInterval() << std::endl;
	printStream << indentString << "\t" << "enabled " << bEnabled << std::endl;
	printStream << indentString << "\t" << "loop " << loop << std::endl;
	printStream << indentString << "\t" << "startTime " << getStartTime() << std::endl;
	printStream << indentString << "\t" << "stopTime " << getStopTime() << std::endl;
}

//////////////////////////////////////////////
//	AudioClip::update
////////////////////////////////////////////////

void TimeSensorNode::update() 
{
	static double currentTime = 0;

	double startTime = getStartTime();
	double stopTime = getStopTime();
	double cycleInterval = getCycleInterval();

	bool bActive	= isActive();
	bool bEnable	= isEnabled();
	bool bLoop		= isLoop();

	if (currentTime == 0)
		currentTime = GetCurrentSystemTime();

	// isActive 
	if (bEnable == false && bActive == true) {
		setIsActive(false);
		sendEvent(getIsActiveField());
		return;
	}

	if (bActive == false && bEnable == true) {
		if (startTime <= currentTime) {
			if (bLoop == true && stopTime <= startTime)
				bActive = true;
			else if (bLoop == false && stopTime <= startTime)
				bActive = true;
			else if (currentTime <= stopTime) {
				if (bLoop == true && startTime < stopTime)
					bActive = true;
				else if	(bLoop == false && startTime < (startTime + cycleInterval) && (startTime + cycleInterval) <= stopTime)
					bActive = true;
				else if (bLoop == false && startTime < stopTime && stopTime < (startTime + cycleInterval))
					bActive = true;
			}
		}
		if (bActive) {
			setIsActive(true);
			sendEvent(getIsActiveField());
			setCycleTime(currentTime);
			sendEvent(getCycleTimeField());
		}
	}

	currentTime = GetCurrentSystemTime();
	
	if (bActive == true && bEnable == true) {
		if (bLoop == true && startTime < stopTime) {
			if (stopTime < currentTime)
				bActive = false;
		}
		else if (bLoop == false && stopTime <= startTime) {
			if (startTime + cycleInterval < currentTime)
				bActive = false;
		}
		else if (bLoop == false && startTime < (startTime + cycleInterval) && (startTime + cycleInterval) <= stopTime) {
			if (startTime + cycleInterval < currentTime)
				bActive = false;
		}
		else if (bLoop == false && startTime < stopTime && stopTime < (startTime + cycleInterval)) {
			if (stopTime < currentTime)
				bActive = false;
		}

		if (bActive == false) {
			setIsActive(false);
			sendEvent(getIsActiveField());
		}
	}

	if (bEnable == false || isActive() == false)
		return;

	// fraction_changed 
	double	fraction = fmod(currentTime - startTime, cycleInterval);
	if (fraction == 0.0 && startTime < currentTime)
		fraction = 1.0;
	else
		fraction /= cycleInterval;
	setFractionChanged((float)fraction);
	sendEvent(getFractionChangedField());

	// cycleTime
	double	cycleTime		= getCycleTime();
	double	cycleEndTime	= cycleTime + cycleInterval;
	while (cycleEndTime < currentTime) {
		setCycleTime(cycleEndTime);
		cycleEndTime += cycleInterval;
		setCycleTime(currentTime);
		sendEvent(getCycleTimeField());
	}

	// time
	setTime(currentTime);
	sendEvent(getTimeField());
}

