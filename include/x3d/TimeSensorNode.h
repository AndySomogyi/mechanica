/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TimeSensorNode.h
*
******************************************************************/

#ifndef _CX3D_TIMESENSORNODE_H_
#define _CX3D_TIMESENSORNODE_H_

#include <time.h>
#ifdef WIN32
#include <sys/timeb.h>
#endif

#include <x3d/SensorNode.h>

namespace CyberX3D {

class TimeSensorNode : public SensorNode {

	SFBool *loopField;
	SFTime *cycleIntervalField;
	SFTime *startTimeField;
	SFTime *stopTimeField;
	SFTime *cycleTimeField;
	SFTime *timeField;
	SFFloat *fractionChangedField;

	SFBool *isPausedField;
	SFTime *pauseTimeField;
	SFTime *resumeTimeField;
	SFTime *elapsedTimeField;
	SFFloat *numLoopsField;

public:

	TimeSensorNode();
	virtual ~TimeSensorNode();

	////////////////////////////////////////////////
	//	Loop
	////////////////////////////////////////////////
	
	SFBool *getLoopField() const;

	void setLoop(bool value);
	void setLoop(int value);
	bool getLoop() const;
	bool isLoop() const;

	////////////////////////////////////////////////
	//	Cyble Interval
	////////////////////////////////////////////////
	
	SFTime *getCycleIntervalField() const;

	void setCycleInterval(double value);
	double getCycleInterval() const; 

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
	//	fraction_changed
	////////////////////////////////////////////////
	
	SFFloat *getFractionChangedField() const;

	void setFractionChanged(float value);
	float getFractionChanged() const;

	////////////////////////////////////////////////
	//	Cycle time
	////////////////////////////////////////////////
	
	SFTime *getCycleTimeField() const;

	void setCycleTime(double value);
	double getCycleTime() const;

	////////////////////////////////////////////////
	//	Time
	////////////////////////////////////////////////
	
	SFTime *getTimeField() const;

	void setTime(double value);
	double getTime() const;

	////////////////////////////////////////////////
	//	IsPaused (X3D)
	////////////////////////////////////////////////
	
	SFBool *getIsPausedField() const;

	void setIsPaused(bool value);
	void setIsPaused(int value);
	bool getIsPaused() const;
	bool isPaused() const;

	////////////////////////////////////////////////
	//	Elapsed time (X3D)
	////////////////////////////////////////////////

	SFTime *getElapsedTimeField() const;
	
	void setElapsedTime(double value);
	double getElapsedTime() const;

	////////////////////////////////////////////////
	//	Pause time (X3D)
	////////////////////////////////////////////////

	SFTime *getPauseTimeField() const;

	void setPauseTime(double value);
	double getPauseTime() const;

	////////////////////////////////////////////////
	//	Resume time (X3D)
	////////////////////////////////////////////////

	SFTime *getResumeTimeField() const;
	
	void setResumeTime(double value);
	double getResumeTime() const;

	////////////////////////////////////////////////
	//	numLoops (X3D)
	////////////////////////////////////////////////

	SFFloat *getNumLoopsField() const;

	void setNumLoops(float value);
	float getNumLoops() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	TimeSensorNode *next() const;
	TimeSensorNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	Virtual functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();
	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

