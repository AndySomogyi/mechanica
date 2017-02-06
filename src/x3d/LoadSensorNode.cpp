/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	LoadSensorNode.cpp
*
*	Revisions:
*		11/15/02
*			- The first revisoin.
*
******************************************************************/

#include <x3d/LoadSensorNode.h>

using namespace CyberX3D;

static const char watchListFieldString[] = "watchList";
static const char isLoadedFieldString[] = "isLoaded";
static const char timeoutFieldString[] = "timeout";
static const char loadTimeFieldString[] = "loadTime";
static const char progressFieldString[] = "progress";

LoadSensorNode::LoadSensorNode() 
{
	setHeaderFlag(false);
	setType(LOADSENSOR_NODE);

	// watchList exposed field
	watchListField = new MFNode();
	addExposedField(watchListFieldString, watchListField);

	// enabled exposed field
	enabledField = new SFBool(true);
	addExposedField(enabledFieldString, enabledField);

	// timeout exposed field
	timeoutField = new SFTime(1.0);
	addExposedField(timeoutFieldString, timeoutField);

	// isActive exposed field
	isActiveField = new SFBool(false);
	addEventOut(isActiveFieldString, isActiveField);

	// loadTime exposed field
	loadTimeField = new SFTime(0.0f);
	addEventOut(loadTimeFieldString, loadTimeField);

	// isLoaded eventOut field
	isLoadedField = new SFBool(false);
	addEventOut(isLoadedFieldString, isLoadedField);

	// progress exposed field
	progressField = new SFFloat(1.0f);
	addEventOut(progressFieldString, progressField);
}

LoadSensorNode::~LoadSensorNode() 
{
}

////////////////////////////////////////////////
//	watchList
////////////////////////////////////////////////

MFNode* LoadSensorNode::getWatchListField() const
{
	if (isInstanceNode() == false)
		return watchListField;
	return (MFNode *)getExposedField(watchListFieldString);
}

////////////////////////////////////////////////
//	Enabled
////////////////////////////////////////////////

SFBool *LoadSensorNode::getEnabledField() const
{
	if (isInstanceNode() == false)
		return enabledField;
	return (SFBool *)getExposedField(enabledFieldString);
}
	
void LoadSensorNode::setEnabled(bool value) 
{
	getEnabledField()->setValue(value);
}

bool LoadSensorNode::getEnabled() const
{
	return getEnabledField()->getValue();
}

bool LoadSensorNode::isEnabled() const
{
	return getEnabled();
}

////////////////////////////////////////////////
//	timeout
////////////////////////////////////////////////

SFTime *LoadSensorNode::getTimeoutField() const
{
	if (isInstanceNode() == false)
		return timeoutField;
	return (SFTime *)getExposedField(timeoutFieldString);
}
	
void LoadSensorNode::setTimeout(double value) 
{
	getTimeoutField()->setValue(value);
}

double LoadSensorNode::getTimeout() const
{
	return getTimeoutField()->getValue();
}

////////////////////////////////////////////////
//	isActive
////////////////////////////////////////////////

SFBool *LoadSensorNode::getIsActiveField() const
{
	if (isInstanceNode() == false)
		return isActiveField;
	return (SFBool *)getEventOut(isActiveFieldString);
}
	
void LoadSensorNode::setIsActive(bool value) 
{
	getIsActiveField()->setValue(value);
}

bool LoadSensorNode::getIsActive() const
{
	return getIsActiveField()->getValue();
}

bool LoadSensorNode::isActive() const
{
	return getIsActive();
}

////////////////////////////////////////////////
//	isLoaded
////////////////////////////////////////////////

SFBool *LoadSensorNode::getIsLoadedField() const
{
	if (isInstanceNode() == false)
		return isLoadedField;
	return (SFBool *)getEventOut(isLoadedFieldString);
}
	
void LoadSensorNode::setIsLoaded(bool value) 
{
	getIsLoadedField()->setValue(value);
}

bool LoadSensorNode::getIsLoaded() const
{
	return getIsLoadedField()->getValue();
}
	
bool LoadSensorNode::isLoaded() const
{
	return getIsLoaded();
}

////////////////////////////////////////////////
//	loadTime
////////////////////////////////////////////////

SFTime *LoadSensorNode::getLoadTimeField() const
{
	if (isInstanceNode() == false)
		return loadTimeField;
	return (SFTime *)getEventOut(loadTimeFieldString);
}
	
void LoadSensorNode::setLoadTime(double value) 
{
	getLoadTimeField()->setValue(value);
}

double LoadSensorNode::getLoadTime() const
{
	return getLoadTimeField()->getValue();
}

////////////////////////////////////////////////
//	progress
////////////////////////////////////////////////

SFFloat *LoadSensorNode::getProgressField() const
{
	if (isInstanceNode() == false)
		return progressField;
	return (SFFloat *)getEventOut(progressFieldString);
}
	
void LoadSensorNode::setProgress(float value) 
{
	getProgressField()->setValue(value);
}

float LoadSensorNode::getProgress() const
{
	return getProgressField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

LoadSensorNode *LoadSensorNode::next() const
{
	return (LoadSensorNode *)Node::next(getType());
}

LoadSensorNode *LoadSensorNode::nextTraversal() const
{
	return (LoadSensorNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	Virtual functions
////////////////////////////////////////////////
	
bool LoadSensorNode::isChildNodeType(Node *node) const
{
	return false;
}

void LoadSensorNode::initialize() 
{
}

void LoadSensorNode::uninitialize() 
{
}

void LoadSensorNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

//////////////////////////////////////////////
//	AudioClip::update
////////////////////////////////////////////////

void LoadSensorNode::update() 
{
}

