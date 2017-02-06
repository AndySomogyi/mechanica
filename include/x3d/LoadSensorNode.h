/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	LoadSensorNode.h
*
******************************************************************/

#ifndef _CX3D_LOADSENSORNODE_H_
#define _CX3D_LOADSENSORNODE_H_

#include <x3d/X3DFields.h>
#include <x3d/NetworkSensorNode.h>

namespace CyberX3D {

class LoadSensorNode : public NetworkSensorNode {

	MFNode* watchListField;
	SFBool *enabledField;
	SFTime *timeoutField;
	SFBool *isActiveField;
	SFBool *isLoadedField;
	SFTime *loadTimeField;
	SFFloat *progressField;

public:

	LoadSensorNode();
	virtual ~LoadSensorNode();

	////////////////////////////////////////////////
	//	watchList
	////////////////////////////////////////////////

	MFNode* getWatchListField() const;

	////////////////////////////////////////////////
	//	Enabled
	////////////////////////////////////////////////
	
	SFBool *getEnabledField() const;

	void setEnabled(bool value);
	bool getEnabled() const;
	bool isEnabled() const;

	////////////////////////////////////////////////
	//	timeout
	////////////////////////////////////////////////
	
	SFTime *getTimeoutField() const;

	void setTimeout(double value);
	double getTimeout() const;

	////////////////////////////////////////////////
	//	isActive
	////////////////////////////////////////////////
	
	SFBool *getIsActiveField() const;

	void setIsActive(bool value);
	bool getIsActive() const;
	bool isActive() const;

	////////////////////////////////////////////////
	//	isLoaded
	////////////////////////////////////////////////
	
	SFBool *getIsLoadedField() const;

	void setIsLoaded(bool value);
	bool getIsLoaded() const;
	bool isLoaded() const;

	////////////////////////////////////////////////
	//	loadTime
	////////////////////////////////////////////////
	
	SFTime *getLoadTimeField() const;

	void setLoadTime(double value);
	double getLoadTime() const;

	////////////////////////////////////////////////
	//	progress
	////////////////////////////////////////////////

	SFFloat *getProgressField() const;

	void setProgress(float value);
	float getProgress() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	LoadSensorNode *next() const;
	LoadSensorNode *nextTraversal() const;

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

