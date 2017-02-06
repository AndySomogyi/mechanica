/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	JScript.h
*
******************************************************************/

#ifndef _CX3D_JSCRIPT_H_
#define _CX3D_JSCRIPT_H_

#if defined(CX3D_SUPPORT_JSAI)

#include <x3d/JNode.h>

namespace CyberX3D {

class ScriptNode;
class Event;

class JScript : public JNode  {

	jclass		mNodeObjectClassID;
	jmethodID	mInitMethodID;
	jmethodID	mSetNameMethodID;

	jmethodID	mAddEventInMethodID;
	jmethodID	mAddEventOutMethodID;
	jmethodID	mAddFieldMethodID;
	jmethodID	mAddExposedFieldMethodID;

	jmethodID	mGetEventInMethodID;
	jmethodID	mGetEventOutMethodID;
	jmethodID	mGetFieldMethodID;
	jmethodID	mGetExposedFieldMethodID;

	jmethodID	mGetNEventInMethodID;
	jmethodID	mGetNEventOutMethodID;
	jmethodID	mGetNFieldsMethodID;
	jmethodID	mGetNExposedFieldsMethodID;

	jmethodID	mInitializeMethodID;
	jmethodID	mShutdownMethodID;
	jmethodID	mProcessEventMethodID;

public:

	JScript(ScriptNode *node);

	virtual ~JScript();

	int			isOK()							{return getNodeObject() ? 1 : 0;}

	////////////////////////////////////////////////
	//	class ID	
	////////////////////////////////////////////////

	jclass		getNodeObjectClassID()			{return mNodeObjectClassID;}
	jmethodID	getInitMethodID()				{return mInitMethodID;}
	jmethodID	getSetNameMethodID()			{return mSetNameMethodID;}

	////////////////////////////////////////////////
	//	method ID
	////////////////////////////////////////////////

	jmethodID	getAddEventInMethodID()			{return mAddEventInMethodID;}
	jmethodID	getAddEventOutMethodID()		{return mAddEventOutMethodID;}
	jmethodID	getAddFieldMethodID()			{return mAddFieldMethodID;}
	jmethodID	getAddExposedFieldMethodID()	{return mAddExposedFieldMethodID;}

	jmethodID	getGetEventInMethodID()			{return mGetEventInMethodID;}
	jmethodID	getGetEventOutMethodID()		{return mGetEventOutMethodID;}
	jmethodID	getGetFieldMethodID()			{return mGetFieldMethodID;}
	jmethodID	getGetExposedFieldMethodID()	{return mGetExposedFieldMethodID;}

	jmethodID	getGetNEventInMethodID()		{return mGetNEventInMethodID;}
	jmethodID	getGetNEventOutMethodID()		{return mGetNEventOutMethodID;}
	jmethodID	getGetNFieldsMethodID()			{return mGetNFieldsMethodID;}
	jmethodID	getGetNExposedFieldsMethodID()	{return mGetNExposedFieldsMethodID;}

	jmethodID	getInitializeMethodID()			{return mInitializeMethodID;}
	jmethodID	getShutdownMethodID()			{return mShutdownMethodID;}
	jmethodID	getProcessEventMethodID()		{return mProcessEventMethodID;}

	////////////////////////////////////////////////
	//	initialize
	////////////////////////////////////////////////

	void		initialize();

	////////////////////////////////////////////////
	//	processEvent
	////////////////////////////////////////////////

	void		processEvent(Event *event);

	////////////////////////////////////////////////
	//	shutdown
	////////////////////////////////////////////////

	void		shutdown();

	////////////////////////////////////////////////
	//	add*	
	////////////////////////////////////////////////

	void addEventOut(Field *field) {
		addFieldObject(getAddEventOutMethodID(), field);
	}

	////////////////////////////////////////////////
	//	set/getValue	
	////////////////////////////////////////////////

	void setValue(Node *node);
	void getValue(Node *node);
};

}

#endif

#endif

