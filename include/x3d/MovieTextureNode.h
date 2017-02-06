/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MovieTextureNode.h
*
******************************************************************/

#ifndef _CX3D_MOVIETEXTURENODE_H_
#define _CX3D_MOVIETEXTURENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Texture2DNode.h>

namespace CyberX3D {

class MovieTextureNode : public Texture2DNode {
	
	MFString *urlField;
	SFBool *loopField;
	SFTime *startTimeField;
	SFTime *stopTimeField;
	SFFloat *speedField;
	SFBool *isActiveField;
	SFTime *durationChangedField;

public:

	MovieTextureNode();
	virtual ~MovieTextureNode();

	////////////////////////////////////////////////
	// Url
	////////////////////////////////////////////////

	MFString *getUrlField() const;

	void addUrl(const char *value);
	int getNUrls() const;
	const char *getUrl(int index) const;
	void setUrl(int index, const char *urlString);

	////////////////////////////////////////////////
	//	Loop
	////////////////////////////////////////////////
	
	SFBool *getLoopField() const;

	void setLoop(bool value);
	void setLoop(int value);
	bool getLoop() const;
	bool isLoop() const;

	////////////////////////////////////////////////
	//	Speed
	////////////////////////////////////////////////
	
	SFFloat *getSpeedField() const;

	void setSpeed(float value);
	float getSpeed() const;

	////////////////////////////////////////////////
	//	Start time
	////////////////////////////////////////////////
	
	SFTime *getStartTimeField() const;

	void setStartTime(double value);
	double getStartTime() const;

	////////////////////////////////////////////////
	//	Stop time
	////////////////////////////////////////////////
	
	SFTime *getStopTimeField() const;

	void setStopTime(double value);
	double getStopTime() const;

	////////////////////////////////////////////////
	//	isActive
	////////////////////////////////////////////////
	
	SFBool *getIsActiveField() const;

	void setIsActive(bool value);
	bool getIsActive() const;
	bool isActive() const;

	////////////////////////////////////////////////
	//	duration_changed
	////////////////////////////////////////////////
	
	SFTime *getDurationChangedField() const;

	void setDurationChanged(double value);
	double getDurationChanged() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	MovieTextureNode *next() const;
	MovieTextureNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	Urlmation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
