/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Event.h
*
******************************************************************/

#ifndef _CX3D_EVENT_H_
#define _CX3D_EVENT_H_

#include <time.h>
#include <x3d/StringUtil.h>
#include <x3d/Field.h>
#include <x3d/SFTime.h>
#include <x3d/JavaVM.h>

namespace CyberX3D {

#if defined(CX3D_SUPPORT_JSAI)
class Event : public JavaVM {
#else
class Event {
#endif

#if defined(CX3D_SUPPORT_JSAI)
	jclass		mEventClass;
#endif

	String		mName;
	double		mTime;
	Field		*mField;

public:

	Event(Field *field);
	Event(const char *name, double time, Field *field);

	void InitializeJavaIDs();

	////////////////////////////////////////////////
	//	Name
	////////////////////////////////////////////////

	void setName(const char *name);
	const char *getName() const;

	////////////////////////////////////////////////
	//	Time
	////////////////////////////////////////////////
	
	void setTimeStamp(double time);
	double getTimeStamp() const;

	////////////////////////////////////////////////
	//	ConstField
	////////////////////////////////////////////////

	void setField(Field *field);
	Field *getField() const;

	////////////////////////////////////////////////
	//	for Java
	////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)

	void setEventClass(jclass eventClass);
	jclass getEventClass();
	jobject toJavaObject();

#endif

};

}

#endif
