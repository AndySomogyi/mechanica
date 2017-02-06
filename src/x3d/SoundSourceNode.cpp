/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SoundSourceNode.cpp
*
*	Revisions:
*
*	12/04/02
*		- The first revision.
*
******************************************************************/

#ifdef WIN32
#include <windows.h>
#endif

#include <x3d/SoundSourceNode.h>

using namespace CyberX3D;

SoundSourceNode::SoundSourceNode() 
{
	// description exposed field
	descriptionField = new SFString("");
	addExposedField(descriptionFieldString, descriptionField);

	// pitch exposed field
	pitchField = new SFFloat(1.0f);
	addExposedField(pitchFieldString, pitchField);

	// url exposed field
	urlField = new MFString();
	addExposedField(urlFieldString, urlField);

	// isActive eventOut field
	isActiveField = new SFBool(false);
	addEventOut(isActiveFieldString, isActiveField);

	// durationChanged eventOut field
	durationChangedField = new SFTime(-1.0f);
	addEventOut(durationFieldString, durationChangedField);
}

SoundSourceNode::~SoundSourceNode() 
{
}

////////////////////////////////////////////////
//	Description
////////////////////////////////////////////////

SFString *SoundSourceNode::getDescriptionField() const
{
	if (isInstanceNode() == false)
		return descriptionField;
	return (SFString *)getExposedField(descriptionFieldString);
}

void SoundSourceNode::setDescription(const char * value) 
{
	getDescriptionField()->setValue(value);
}

const char *SoundSourceNode::getDescription() const
{
	return getDescriptionField()->getValue();
}

////////////////////////////////////////////////
//	Pitch
////////////////////////////////////////////////

SFFloat *SoundSourceNode::getPitchField() const
{
	if (isInstanceNode() == false)
		return pitchField;
	return (SFFloat *)getExposedField(pitchFieldString);
}
	
void SoundSourceNode::setPitch(float value) 
{
	getPitchField()->setValue(value);
}

float SoundSourceNode::getPitch() const
{
	return getPitchField()->getValue();
}

////////////////////////////////////////////////
//	isActive
////////////////////////////////////////////////

SFBool *SoundSourceNode::getIsActiveField() const
{
	if (isInstanceNode() == false)
		return isActiveField;
	return (SFBool *)getEventOut(isActiveFieldString);
}
	
void SoundSourceNode::setIsActive(bool  value) 
{
	getIsActiveField()->setValue(value);
}

bool SoundSourceNode::getIsActive() const
{
	return getIsActiveField()->getValue();
}

bool SoundSourceNode::isActive() const
{
	return getIsActiveField()->getValue();
}

////////////////////////////////////////////////
//	duration_changed
////////////////////////////////////////////////

SFTime *SoundSourceNode::getDurationChangedField() const
{
	if (isInstanceNode() == false)
		return durationChangedField;
	return (SFTime *)getEventOut(durationFieldString);
}
	
void SoundSourceNode::setDurationChanged(double value) 
{
	getDurationChangedField()->setValue(value);
}

double SoundSourceNode::getDurationChanged() const
{
	return getDurationChangedField()->getValue();
}

////////////////////////////////////////////////
// Url
////////////////////////////////////////////////

MFString *SoundSourceNode::getUrlField() const
{
	if (isInstanceNode() == false)
		return urlField;
	return (MFString *)getExposedField(urlFieldString);
}

void SoundSourceNode::addUrl(const char * value) 
{
	getUrlField()->addValue(value);
}

int SoundSourceNode::getNUrls() const
{
	return getUrlField()->getSize();
}

const char *SoundSourceNode::getUrl(int index) const
{
	return getUrlField()->get1Value(index);
}

void SoundSourceNode::setUrl(int index, const char *urlString) 
{
	getUrlField()->set1Value(index, urlString);
}
