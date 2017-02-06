/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	AudioClipNode.h
*
******************************************************************/

#ifndef _CX3D_AUDIOCLLIPNODE_H_
#define _CX3D_AUDIOCLLIPNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/SoundSourceNode.h>

namespace CyberX3D {

class AudioClipNode : public SoundSourceNode 
{
	SFBool *loopField;
	SFTime *startTimeField;
	SFTime *stopTimeField;

	double	mCurrentTime;

public:

	AudioClipNode();
	virtual ~AudioClipNode();

	////////////////////////////////////////////////
	//	Loop
	////////////////////////////////////////////////

	SFBool *getLoopField() const;
	
	void setLoop(bool value);
	void setLoop(int value);
	bool getLoop() const;
	bool isLoop() const;

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
	//	List
	////////////////////////////////////////////////

	AudioClipNode *next() const;
	AudioClipNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	Virutual functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	void outputContext(std::ostream &printStream, const char *indentString) const;

	////////////////////////////////////////////////
	//	Time
	////////////////////////////////////////////////

	void setCurrentTime(double time);
	double getCurrentTime() const;
};

}

#endif
